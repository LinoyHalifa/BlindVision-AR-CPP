// Build:
// g++ BlindVisionAR.cpp -o BlindVisionAR -std=c++17 `pkg-config --cflags --libs opencv4`
// On Windows with MSVC: add opencv includes/libs and link ole32.lib, sapi.lib

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#ifdef _WIN32
#include <sapi.h>         // SAPI 5
#include <comdef.h>
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "sapi.lib")
#endif

using namespace cv;
using namespace std;

// ====== Config ======
static const float CONF_THRESHOLD = 0.50f; // confidence threshold
static const float NMS_THRESHOLD  = 0.45f; // NMS IoU threshold
static const int   INPUT_W = 640;          // YOLO input width
static const int   INPUT_H = 640;          // YOLO input height

// COCO class names (80)
static const vector<string> COCO_NAMES = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog",
    "horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
    "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich",
    "orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote",
    "keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book",
    "clock","vase","scissors","teddy bear","hair drier","toothbrush"
};

#ifdef _WIN32
// Simple TTS wrapper using SAPI
class WinTTS {
public:
    WinTTS() : pVoice(nullptr), comOk(false) {
        HRESULT hr = CoInitialize(NULL);
        comOk = SUCCEEDED(hr) || hr == RPC_E_CHANGED_MODE;
        if (comOk) {
            CoCreateInstance(CLSID_SpVoice, NULL, CLSCTX_ALL, IID_ISpVoice, (void**)&pVoice);
        }
    }
    ~WinTTS() {
        if (pVoice) pVoice->Release();
        if (comOk) CoUninitialize();
    }
    void say(const wstring& text) {
        if (pVoice) pVoice->Speak(text.c_str(), SPF_ASYNC, NULL);
    }
private:
    ISpVoice* pVoice;
    bool comOk;
};
#endif

struct Detection {
    Rect box;
    int class_id;
    float confidence;
};

// Convert (cx,cy,w,h) to left-top-width-height rect
static inline Rect toRect(float cx, float cy, float w, float h, float xFactor, float yFactor, int xPad, int yPad) {
    int left   = (int)((cx - 0.5f * w) * xFactor - xPad);
    int top    = (int)((cy - 0.5f * h) * yFactor - yPad);
    int width  = (int)(w * xFactor);
    int height = (int)(h * yFactor);
    return Rect(left, top, width, height);
}

// Parse YOLOv8 output (supports the common OpenCV DNN shape: [1, 84, N])
static vector<Detection> parseDetections(const Mat& out, const Size& frameSize) {
    vector<Detection> dets;
    // Expected shape: 3 dims [1 x 84 x N]
    if (out.dims != 3) {
        cerr << "Unexpected output dims: " << out.dims << endl;
        return dets;
    }

    int dims1 = out.size[1]; // 84 = 4 box + 80 classes
    int dims2 = out.size[2]; // N = number of predictions (e.g., 8400)
    if (dims1 < 5) {
        cerr << "Unexpected output shape." << endl;
        return dets;
    }

    // Wrap as a 84xN matrix for easier indexing
    Mat outMat(dims1, dims2, CV_32F, (void*)out.ptr<float>());

    // Scaling from 640x640 letterbox back to frame
    float xFactor = (float)frameSize.width  / INPUT_W;
    float yFactor = (float)frameSize.height / INPUT_H;

    // Collect raw boxes + confidence
    vector<Rect> boxes;
    vector<float> scores;
    vector<int> classIds;

    for (int i = 0; i < dims2; ++i) {
        float cx = outMat.at<float>(0, i);
        float cy = outMat.at<float>(1, i);
        float w  = outMat.at<float>(2, i);
        float h  = outMat.at<float>(3, i);

        // class scores start at index 4
        Range scoreRange(4, dims1);
        Mat scoresRow = outMat.rowRange(scoreRange).col(i);
        Point classIdPoint;
        double maxScore;
        minMaxLoc(scoresRow, 0, &maxScore, 0, &classIdPoint);

        float conf = (float)maxScore;
        if (conf >= CONF_THRESHOLD) {
            int cls = classIdPoint.y; // since scoresRow is a column, y is index
            Rect r = toRect(cx, cy, w, h, xFactor, yFactor, 0, 0);

            // keep only boxes that intersect the frame
            Rect valid = r & Rect(0, 0, frameSize.width, frameSize.height);
            if (valid.width > 0 && valid.height > 0) {
                boxes.push_back(valid);
                scores.push_back(conf);
                classIds.push_back(cls);
            }
        }
    }

    // NMS
    vector<int> indices;
    dnn::NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD, indices);

    for (int idx : indices) {
        Detection d;
        d.box = boxes[idx];
        d.class_id = classIds[idx];
        d.confidence = scores[idx];
        dets.push_back(d);
    }
    return dets;
}

static string directionFromBox(const Rect& box, int frameWidth) {
    float center_x = box.x + box.width * 0.5f;
    if (center_x < frameWidth / 3.0f) return "on your left";
    if (center_x > 2.0f * frameWidth / 3.0f) return "on your right";
    return "in front";
}

int main() {
    // Load model
    const string modelPath = "yolov8n.onnx"; // <-- set path if needed
    dnn::Net net = dnn::readNetFromONNX(modelPath);
    if (net.empty()) {
        cerr << "Failed to load model: " << modelPath << endl;
        return 1;
    }
    // Prefer CUDA if available (optional):
    // net.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
    // net.setPreferableTarget(dnn::DNN_TARGET_CUDA);

#ifdef _WIN32
    WinTTS tts;
    auto say = [&](const string& s) {
        cout << "[AUDIO] " << s << endl;
        // Convert to wide string for SAPI (UTF-16)
        wstring ws(s.begin(), s.end());
        tts.say(ws);
    };
#else
    auto say = [&](const string& s) {
        cout << "[AUDIO] " << s << endl;
    };
#endif

    // Open camera
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Camera Is Not Connect" << endl;
        return 1;
    } else {
        cout << "Camera Is Connect" << endl;
    }

    // Window
    const string kWin = "BlindVision AR";
    namedWindow(kWin, WINDOW_AUTOSIZE);

    while (true) {
        Mat frame;
        if (!cap.read(frame) || frame.empty()) break;

        // Preprocess
        Mat blob = dnn::blobFromImage(frame, 1.0/255.0, Size(INPUT_W, INPUT_H), Scalar(), true, false);
        net.setInput(blob);

        // Forward
        Mat out = net.forward(); // typically returns [1,84,N]
        vector<Detection> dets = parseDetections(out, frame.size());

        // Draw detections
        Mat annotated = frame.clone();
        for (const auto& d : dets) {
            rectangle(annotated, d.box, Scalar(0, 255, 0), 2);
            string label = COCO_NAMES[d.class_id] + " " + to_string((int)round(d.confidence * 100)) + "%";
            int base; Size tsize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
            int y = max(d.box.y, tsize.height + 5);
            rectangle(annotated, Point(d.box.x, y - tsize.height - 6),
                      Point(d.box.x + tsize.width + 6, y + 4), Scalar(0, 0, 0), FILLED);
            putText(annotated, label, Point(d.box.x + 3, y - 3), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
        }

        // Show
        imshow(kWin, annotated);

        // Speak only one object per frame (highest confidence after NMS is first)
        if (!dets.empty()) {
            const auto& best = dets[0];
            string dir = directionFromBox(best.box, frame.cols);
            string clsName = (best.class_id >= 0 && best.class_id < (int)COCO_NAMES.size())
                             ? COCO_NAMES[best.class_id] : "object";
            string msg = clsName + " " + dir;
            say(msg);
        }

        // Exit on 'q' or ESC
        int key = waitKey(1);
        if (key == 'q' || key == 27) break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
