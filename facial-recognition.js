// Facial Recognition System
class FacialRecognitionSystem {
  constructor() {
    this.video = document.getElementById('webcam');
    this.canvas = document.getElementById('canvas');
    this.startBtn = document.getElementById('startBtn');
    this.stopBtn = document.getElementById('stopBtn');
    this.continueBtn = document.getElementById('continueBtn');
    this.statusIndicator = document.getElementById('status-indicator');
    this.resultsSection = document.getElementById('resultsSection');
    this.detectionResults = document.getElementById('detectionResults');

    this.stream = null;
    this.modelsLoaded = false;
    this.modelLoading = true;
    this.faceDetector = null;
    this.useFaceApi = false;
    this.detectionInterval = null;
    this.cameraWrapper = document.querySelector('.camera-wrapper');
    this.autoConfirmed = false;
    this.smileDetected = false;
    this.blinkDetected = false;
    this.blinkInProgress = false;
    this.blinkCount = 0;

    this.initEventListeners();
    if (this.startBtn) {
      this.startBtn.disabled = true;
    }
    this.loadModels();
  }

  initEventListeners() {
    this.startBtn.addEventListener('click', () => this.startCamera());
    this.stopBtn.addEventListener('click', () => this.stopCamera());
  }

  async loadModels() {
    if ('faceapi' in window) {
      try {
        const MODEL_URL = 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights';
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
          faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
          faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
        ]);
        this.modelsLoaded = true;
        this.modelLoading = false;
        this.useFaceApi = true;
        console.log('✓ face-api.js models loaded successfully');
        if (this.startBtn) {
          this.startBtn.disabled = false;
        }
      } catch (error) {
        console.error('Error loading face-api.js models:', error);
        this.modelsLoaded = false;
        this.modelLoading = false;
        this.updateStatus('Face detection unavailable', 'error');
      }
      return;
    }

    if ('FaceDetector' in window) {
      this.faceDetector = new window.FaceDetector({ fastMode: true, maxDetectedFaces: 2 });
      this.modelsLoaded = true;
      this.modelLoading = false;
      console.log('✓ Browser FaceDetector API is available.');
      if (this.startBtn) {
        this.startBtn.disabled = false;
      }
      return;
    }

    this.modelsLoaded = false;
    this.modelLoading = false;
    this.updateStatus('Face detection unavailable', 'error');
  }

  async startCamera() {
    if (this.modelLoading) {
      this.updateStatus('Loading face detection model...', 'active');
      alert('Please wait a moment while face detection models are loading.');
      return;
    }

    if (!this.modelsLoaded) {
      this.updateStatus('Face detection unavailable', 'error');
      alert('Face detection is unavailable in this browser or failed to load. Please use a supported browser such as Chrome or Edge and try again.');
      return;
    }

    try {
      this.updateStatus('Initializing...', 'active');

      const constraints = {
        audio: false,
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        }
      };

      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
      this.video.srcObject = this.stream;

      this.video.onloadedmetadata = () => {
        this.video.play();
        this.updateStatus('Camera Active', 'active');
        this.startBtn.disabled = true;
        this.stopBtn.disabled = false;
        if (this.continueBtn) {
          this.continueBtn.style.display = 'none';
        }
        this.autoConfirmed = false;
        this.smileDetected = false;
        this.blinkDetected = false;
        this.blinkInProgress = false;
        this.blinkCount = 0;
        this.detectFaceContinuous();
      };
    } catch (error) {
      console.error('Error accessing camera:', error);
      alert('Unable to access camera. Please check permissions.');
      this.updateStatus('Camera Error', 'error');
    }
  }

  stopCamera() {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.video.srcObject = null;
      this.updateStatus('Ready', 'idle');
      this.startBtn.disabled = false;
      this.stopBtn.disabled = true;
      this.resultsSection.style.display = 'none';
      if (this.detectionInterval) {
        clearInterval(this.detectionInterval);
        this.detectionInterval = null;
      }
    }
  }

  async detectFaceContinuous() {
    if (!this.stream || !this.modelsLoaded) return;

    if (this.detectionInterval) {
      clearInterval(this.detectionInterval);
    }

    this.detectionInterval = setInterval(async () => {
      try {
        const faceData = await this.detectFaceData(this.video);
        let face = null;

        if (Array.isArray(faceData)) {
          face = faceData[0];
        } else {
          face = faceData;
        }

        if (face) {
          const box = this.getFaceBox(face);
          const message = this.evaluateFacePosition(box);

          if (this.useFaceApi) {
            const liveness = this.evaluateLiveness(face);
            this.updateStatus(liveness.ok ? 'Smile and blink confirmed — verifying...' : liveness.message, liveness.ok ? 'detected' : 'active');
            if (this.cameraWrapper) {
              this.cameraWrapper.classList.toggle('aligned', message.includes('ready'));
            }
            if (liveness.ok && !this.autoConfirmed) {
              this.confirmVerification();
            }
          } else {
            this.updateStatus(message, message.includes('ready') ? 'detected' : 'active');
            if (this.cameraWrapper) {
              this.cameraWrapper.classList.toggle('aligned', message.includes('ready'));
            }
          }
        } else {
          this.updateStatus('No Face Detected', 'active');
          if (this.cameraWrapper) {
            this.cameraWrapper.classList.remove('aligned');
          }
        }
      } catch (error) {
        console.error('Continuous detection error:', error);
      }
    }, 200);
  }

  async detectFaceData(source) {
    if (this.useFaceApi) {
      return await faceapi.detectSingleFace(source, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceExpressions();
    }

    if (this.faceDetector) {
      return await this.faceDetector.detect(source);
    }

    return null;
  }

  async detectFaces(source) {
    if (this.faceDetector) {
      return await this.faceDetector.detect(source);
    }

    if (this.useFaceApi) {
      const result = await faceapi.detectAllFaces(source, new faceapi.TinyFaceDetectorOptions());
      return result;
    }

    return [];
  }

  getFaceBox(face) {
    return face.detection?.box || face.boundingBox || face.box || null;
  }

  evaluateFacePosition(box) {
    if (!box || !this.video) return 'Please keep your face in view';
    const vw = this.video.videoWidth;
    const vh = this.video.videoHeight;
    const centerX = box.x + box.width / 2;
    const centerY = box.y + box.height / 2;
    const horizontalOffset = Math.abs(centerX - vw / 2) / vw;
    const verticalOffset = Math.abs(centerY - vh / 2) / vh;
    const areaRatio = (box.width * box.height) / (vw * vh);
    if (areaRatio < 0.08) return 'Move closer to the camera';
    if (areaRatio > 0.35) return 'Move back slightly';
    if (horizontalOffset > 0.15 || verticalOffset > 0.15) return 'Center your face inside the circle';
    return 'Face aligned — ready to capture';
  }

  calculateEyeAspectRatio(eye) {
    if (!eye || eye.length < 6) return 0;
    const distance = (p1, p2) => Math.hypot(p2.x - p1.x, p2.y - p1.y);
    const a = distance(eye[1], eye[5]);
    const b = distance(eye[2], eye[4]);
    const c = distance(eye[0], eye[3]);
    return c === 0 ? 0 : (a + b) / (2.0 * c);
  }

  evaluateLiveness(face) {
    const expressions = face.expressions || {};
    const landmarks = face.landmarks;
    let smileDetected = false;
    let blinkDetected = this.blinkDetected;
    let message = 'Face detected — please smile and blink to confirm.';

    console.log('Expressions:', expressions);
    if (expressions.happy >= 0.3) {
      smileDetected = true;
      console.log('Smile detected');
    }

    if (landmarks) {
      const leftEye = landmarks.getLeftEye();
      const rightEye = landmarks.getRightEye();
      const leftEAR = this.calculateEyeAspectRatio(leftEye);
      const rightEAR = this.calculateEyeAspectRatio(rightEye);
      console.log('Left EAR:', leftEAR, 'Right EAR:', rightEAR);
      const eyesClosed = leftEAR < 0.27 && rightEAR < 0.27;

      if (eyesClosed) {
        if (!this.blinkInProgress) {
          this.blinkCount += 1;
          this.blinkInProgress = true;
          console.log('Blink detected, count:', this.blinkCount);
        }
      } else {
        this.blinkInProgress = false;
      }

      if (this.blinkCount >= 1) {
        blinkDetected = true;
        this.blinkDetected = true;
        console.log('Blink confirmed');
      }
    }

    if (!smileDetected && !blinkDetected) {
      message = 'Please smile and blink in front of the camera.';
    } else if (smileDetected && !blinkDetected) {
      message = 'Smile detected — blink once to complete verification.';
    } else if (!smileDetected && blinkDetected) {
      message = 'Blink detected — please smile now to complete verification.';
    } else if (smileDetected && blinkDetected) {
      message = 'Smile and blink confirmed — ready to capture.';
    }

    console.log('Liveness result:', { smileDetected, blinkDetected, ok: smileDetected && blinkDetected });
    return {
      smileDetected,
      blinkDetected,
      message,
      ok: smileDetected && blinkDetected
    };
  }

  async captureAndVerify() {
    if (!this.modelsLoaded) {
      this.updateStatus('Face detection unavailable', 'error');
      this.detectionResults.innerHTML = '<p>❌ Face detection could not run because the model failed to load.</p>';
      this.resultsSection.style.display = 'block';
      return;
    }

    try {
      const ctx = this.canvas.getContext('2d');
      this.canvas.width = this.video.videoWidth;
      this.canvas.height = this.video.videoHeight;
      ctx.drawImage(this.video, 0, 0);

      const faceData = await this.detectFaceData(this.video);
      this.resultsSection.style.display = 'block';

      if (!faceData) {
        this.detectionResults.innerHTML = '<p>❌ No face detected. Please position your face clearly and try again.</p>';
        this.updateStatus('No Face Found', 'active');
        return;
      }

      const face = Array.isArray(faceData) ? faceData[0] : faceData;
      if (!face) {
        this.detectionResults.innerHTML = '<p>❌ No face detected. Please position your face clearly and try again.</p>';
        this.updateStatus('No Face Found', 'active');
        return;
      }

      if (!this.useFaceApi) {
        this.detectionResults.innerHTML = '<p>⚠️ Liveness verification requires a modern browser. Please use Chrome or Edge.</p>';
        this.updateStatus('Browser Unsupported', 'error');
        return;
      }

      const liveness = this.evaluateLiveness(face);
      if (!liveness.ok) {
        this.detectionResults.innerHTML = `<p>❌ ${liveness.message}</p><p>Please hold still and try again.</p>`;
        this.updateStatus('Liveness Not Confirmed', 'active');
        return;
      }

      const resultsHTML = `
        <p><strong>✓ Face Successfully Captured</strong></p>
        <p><strong>Identity Confirmed.</strong></p>
        <p style="color: #7EBD6C; margin-top: 15px;"><strong>✓ Smile and blink verified.</strong></p>
      `;

      this.detectionResults.innerHTML = resultsHTML;
      this.updateStatus('Verification Complete', 'verified');
      if (this.continueBtn) {
        this.continueBtn.style.display = 'inline-flex';
      }
      localStorage.setItem('faceIdConfirmed', 'true');
      this.captureBtn.disabled = true;
    } catch (error) {
      console.error('Verification error:', error);
      this.detectionResults.innerHTML = '<p>❌ Error during verification. Please try again.</p>';
      this.updateStatus('Error', 'error');
    }
  }

  confirmVerification() {
    this.autoConfirmed = true;
    if (this.detectionInterval) {
      clearInterval(this.detectionInterval);
      this.detectionInterval = null;
    }

    // Create overlay modal
    const overlay = document.createElement('div');
    overlay.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.7);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 9999;
    `;

    overlay.innerHTML = `
      <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 20px; max-width: 400px; background: white; padding: 40px 20px; border-radius: 16px; box-shadow: 0 10px 40px rgba(0,0,0,0.3);">
        <div style="width: 100px; height: 100px; border: 4px solid #7EBD6C; border-radius: 50%; display: flex; align-items: center; justify-content: center; background: rgba(126, 189, 108, 0.15);">
          <span style="font-size: 60px; color: #7EBD6C;">✓</span>
        </div>
        <div style="text-align: center;">
          <h2 style="font-size: 28px; color: #2c3e50; margin: 15px 0; font-weight: 700;">Identity Confirmed</h2>
          <p style="font-size: 15px; color: #666666; margin: 10px 0 20px 0; line-height: 1.6;">Your face has been verified with smile and blink detection. You may now proceed to change your pin number.</p>
        </div>
        <button id="confirmOkBtn" style="padding: 12px 30px; background-color: #7EBD6C; color: white; border: none; border-radius: 8px; font-size: 16px; font-weight: 600; cursor: pointer; margin-top: 10px; transition: background-color 0.2s;">OK</button>
      </div>
    `;

    document.body.appendChild(overlay);

    document.getElementById('confirmOkBtn').addEventListener('click', () => {
      overlay.remove();
      window.location.href = 'change-pin.html';
    });

    this.updateStatus('Identity Confirmed', 'verified');
    localStorage.setItem('faceIdConfirmed', 'true');
  }

  updateStatus(text, state) {
    this.statusIndicator.textContent = text;
    this.statusIndicator.className = `status-${state === 'verified' ? 'detected' : state}`;
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    new FacialRecognitionSystem();
  });
} else {
  new FacialRecognitionSystem();
}
