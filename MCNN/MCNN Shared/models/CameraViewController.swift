//
//  CameraViewController.swift
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/5/4.
//  Code Reference: https://www.youtube.com/watch?v=Zv4cJf5qdu0
//

import UIKit
import AVFoundation

class CameraViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    public typealias DataType = Float32
    private static let N_INPUT_CHANNELS = 3
    
    let captureSession = AVCaptureSession()
    var previewLayer: CALayer!
    var captureDevice: AVCaptureDevice!
    var detecting: Bool = false
    let vggNetwork = VGG11(gpu: true)
    
    override func viewDidLoad() {
        super.viewDidLoad()
        prepareCamera()
    }

    func prepareCamera() {
        captureSession.sessionPreset = AVCaptureSession.Preset.photo
        let availableDevices = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera], mediaType: AVMediaType.video,
            position: .back).devices
        captureDevice = availableDevices.first
        beginSession()
    }
    
    func beginSession () {
        do {
            let captureDeviceInput = try AVCaptureDeviceInput(device: captureDevice)
            captureSession.addInput(captureDeviceInput)
        } catch {
            print(error.localizedDescription)
        }
        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        self.previewLayer = previewLayer
        self.view.layer.addSublayer(self.previewLayer)
        self.previewLayer.frame = self.view.layer.frame
        captureSession.startRunning()
        
        let dataOutput = AVCaptureVideoDataOutput()
        dataOutput.videoSettings = [
            (kCVPixelBufferPixelFormatTypeKey as String): NSNumber(value: kCVPixelFormatType_32RGBA)]
        
        dataOutput.alwaysDiscardsLateVideoFrames = true
        
        if captureSession.canAddOutput(dataOutput) {
            captureSession.addOutput(dataOutput)
        }
        
        captureSession.commitConfiguration()
        
        let queue = DispatchQueue(label: "cameraDetectQueue")
        dataOutput.setSampleBufferDelegate(self, queue: queue)
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        if detecting {
            return
        }
        if let image = self.getImageFromSampleBuffer(buffer: sampleBuffer) {
            detecting = true

            DispatchQueue.main.async {
                let imgTensor: Tensor<DataType> = CameraViewController.imageToTensor(image)
                /*
                print([imgTensor.getData(idx: [0, 0, 500, 700]),
                       imgTensor.getData(idx: [0, 1, 500, 700]),
                       imgTensor.getData(idx: [0, 2, 500, 700])])
                 */
                imgTensor.copyToGPU()
                let forwardResult: Tensor<DataType> = self.vggNetwork.forward(input: imgTensor)
                forwardResult.copyToCPU()
                forwardResult.printData()
                // self.detecting = false
            }
        }
    }
    
    func getImageFromSampleBuffer(buffer: CMSampleBuffer) -> UIImage? {
        if let pixelBuffer = CMSampleBufferGetImageBuffer(buffer) {
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            let context = CIContext()
            
            let imageRect = CGRect(x: 0, y: 0, width: CVPixelBufferGetWidth(pixelBuffer),
                                   height: CVPixelBufferGetHeight(pixelBuffer))
            if let image = context.createCGImage(ciImage, from: imageRect) {
                return UIImage(cgImage: image, scale: UIScreen.main.scale, orientation: .right)
            }
        }
        return nil
    }

    private static func imageToTensor(_ inputImage: UIImage) -> Tensor<DataType> {
        let image: UIImage = ResizeImage(
            image: inputImage, targetSize: CGSize(width: 32, height: 32))

        guard let cgImage = image.cgImage,
              let data = cgImage.dataProvider?.data,
              let bytes = CFDataGetBytePtr(data) else {
            fatalError("Couldn't access image data")
        }
        assert(cgImage.colorSpace?.model == .rgb)

        let outputHeight = cgImage.height
        let outputWidth = cgImage.width
        
        var outputArray: [DataType] = Array(repeating: 0,
                                            count: N_INPUT_CHANNELS * outputHeight * outputWidth)

        let bytesPerPixel = cgImage.bitsPerPixel / cgImage.bitsPerComponent
        for y in 0 ..< outputHeight {
            for x in 0 ..< outputWidth {
                let offset = (y * cgImage.bytesPerRow) + (x * bytesPerPixel)
                for imgChannel in 0 ..< N_INPUT_CHANNELS {
                    outputArray[
                        imgChannel * (outputHeight * outputWidth) +
                        y * (outputWidth) +
                        x
                    ] = DataType(bytes[offset + imgChannel]) / 256
                }
            }
        }
        return Tensor<DataType>(shape: [1, N_INPUT_CHANNELS, outputHeight, outputWidth], data: outputArray)
    }
    
    private static func ResizeImage(image: UIImage, targetSize: CGSize) -> UIImage {
        /*
        let size = image.size

        let widthRatio  = targetSize.width  / image.size.width
        let heightRatio = targetSize.height / image.size.height

        // Figure out what our orientation is, and use that to form the rectangle
        var newSize: CGSize

        if(widthRatio > heightRatio) {
            newSize = CGSize(width: size.width * heightRatio, height: size.height * heightRatio)
        } else {
            newSize = CGSize(width: size.width * widthRatio,  height: size.height * widthRatio)
        }
        */
        let newSize: CGSize = CGSize(width: targetSize.width,  height: targetSize.height)

        // This is the rect that we've calculated out and this is what is actually used below
        let rect = CGRect(x: 0, y: 0, width: newSize.width, height: newSize.height)

        // Actually do the resizing to the rect using the ImageContext stuff
        UIGraphicsBeginImageContextWithOptions(newSize, false, 1.0)
        image.draw(in: rect)
        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()

        return newImage!
    }
    
    func stopCaptureSession() {
        self.captureSession.stopRunning()
        
        if let inputs = captureSession.inputs as? [AVCaptureDeviceInput] {
            for input in inputs {
                self.captureSession.removeInput(input)
            }
        }
    }
    
    
}
