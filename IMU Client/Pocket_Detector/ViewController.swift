//
//  ViewController.swift
//  SensorLogger
//
//  Created by Vicky Liu on 2/25/22.
//

import CoreMotion
import UIKit
import WatchConnectivity
import NearbyInteraction
import CoreBluetooth

class ViewController: UIViewController {
    
    @IBOutlet weak var socketStatusLabel: UILabel!
    @IBOutlet weak var phoneStatusLabel: UILabel!
    
    @IBOutlet weak var phoneSamplingRatePickerView: UIPickerView!
    
    @IBOutlet weak var socketIPField: UITextField!
    @IBOutlet weak var socketPortField: UITextField!
    @IBOutlet weak var deviceIdentifierField: UITextField!
    
    let nToMeasureFrequency = 50
    var frequencies: [Double] = []
    
    // phone
    var phoneCnt = 0
    var phonePrevTime: TimeInterval = NSDate().timeIntervalSince1970
    var phoneSetFrequency: Double? = nil
    var phoneMeasuredFrequency: Double? = nil
    
    // managers
    var phoneMotionManager: CMMotionManager!
    var phoneQueue = OperationQueue()
    
    // socket
    var socketClient: SocketClient?
    
    // nearby
    var niSession: NISession?
    var iPhoneTokenData: Data?
    
    // BLE
    var peripheralManager: CBPeripheralManager!
    var tokenService: CBMutableService?
    var iPhoneTokenCharacteristic: CBMutableCharacteristic?
    let tokenServiceUUID: CBUUID = CBUUID(string:"2AC0B600-7C0C-4C9D-AB71-072AE2037107")
    let iPhoneTokenCharacteristicUUID: CBUUID = CBUUID(string:"2AC0B602-7C0C-4C9D-AB71-072AE2037107")
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        if (WCSession.isSupported()) {
            let session = WCSession.default
            session.delegate = self
            session.activate()
        }
        
        stride(from: 1.0, to: 61.0, by: 1.0).forEach{ v in
            frequencies.append(v)
        }
        phoneSamplingRatePickerView.tag = 1
        
        
        [socketIPField, socketPortField, deviceIdentifierField].forEach { field in
            field?.delegate = self
        }
        [phoneSamplingRatePickerView].forEach { picker in
            picker?.delegate = self
            picker?.dataSource = self
        }
        
        self.hideKeyboardWhenTapped()
        
        // proximity
        /*UIDevice.current.isProximityMonitoringEnabled = true
        let notificationName = Notification.Name(rawValue: "UIDeviceProximityStateDidChangeNotification")
        NotificationCenter.default.addObserver(self, selector: #selector(proximityStateDidChange), name: notificationName, object: nil)*/
        
        // background
        NotificationCenter.default.addObserver(self, selector: #selector(didEnterBackground),
                                    name: UIScene.didEnterBackgroundNotification, object: nil)
        NotificationCenter.default.addObserver(self, selector: #selector(willEnterForeground),
                                     name: UIScene.willEnterForegroundNotification, object: nil)
        
        //startNIprocess()
        
        socketIPField.text = "192.168.0.129"
        socketPortField.text = "8001"
        deviceIdentifierField.text = "phone"
        
        phoneSamplingRatePickerView.selectRow(23, inComponent: 0, animated: false)
    }
    
    /*@objc func proximityStateDidChange(notification: Notification) {
      if let device = notification.object as? UIDevice {
        
        let currentProximityState = device.proximityState
          guard let socketClient = socketClient else {
              return
          }
          socketClient.send(text: "proximity: \(currentProximityState)")
      }
    }*/
    
    @objc func didEnterBackground(_ notification: Notification) {
        guard let socketClient = socketClient else {
            return
        }
        socketClient.send(text: "app: background")
    }
    
    @objc func willEnterForeground(_ notification: Notification) {
        
        // nearby
        /*startNIprocess()
        
        if (WCSession.default.isReachable) {
            do {
                try WCSession.default.updateApplicationContext(["command": "ni"])
                print("sent ni command to watch")
            }
            catch {
                print(error)
            }
        } else {
            print("watch is not reachable")
        }*/
        
        guard let socketClient = socketClient else {
            return
        }
        socketClient.send(text: "app: foreground")
 
    }
    
    /*private func startNIprocess() {
        guard NISession.isSupported else {
            print("This device doesn't support Nearby Interaction.")
            return
        }
        niSession = NISession()
        niSession?.delegate = self
        guard let token = niSession?.discoveryToken else {
            return
        }
        iPhoneTokenData = try! NSKeyedArchiver.archivedData(withRootObject: token, requiringSecureCoding: true)
        
        // BLE
        peripheralManager = CBPeripheralManager(delegate: self, queue: nil)
        iPhoneTokenCharacteristic = CBMutableCharacteristic(type: iPhoneTokenCharacteristicUUID, properties: [.read], value: iPhoneTokenData, permissions: [.readable])
        tokenService = CBMutableService(type: tokenServiceUUID, primary: true)
        tokenService?.characteristics = [iPhoneTokenCharacteristic!]
    }*/
    
    
    private func updateSocketStatusLabel(status: Bool) {
        DispatchQueue.main.async {
            if status {
                self.socketStatusLabel.text = "socket ready"
            } else {
                self.socketStatusLabel.text = "socket not ready"
            }
        }
    }
    
    @IBAction func createSocketConnection() {
        let ip = socketIPField.text as String? ?? "0.0.0.0"
        let port = UInt16(socketPortField.text as String? ?? "0") ?? 8000
        let deviceIdentifier = deviceIdentifierField.text as String? ?? "sample"
    
        print("try to connect \(ip):\(port) with device id \(deviceIdentifier)")
        socketClient = SocketClient(ip: ip, portInt: port, deviceID: deviceIdentifier) { (status) in
            self.updateSocketStatusLabel(status: status)
        }
    }
    
    @IBAction func restartSocketConnection() {
        guard let socketClient = socketClient else {
            return
        }
        socketClient.restart() { (status) in
            self.updateSocketStatusLabel(status: status)
        }
        socketClient.send(text: "restarted")
    }
    
    @IBAction func stopSocketConnection() {
        guard let socketClient = socketClient else {
            return
        }
        socketClient.stop()
    }
    
    @IBAction func startPhoneMotion() {
        
        phoneMotionManager = CMMotionManager()
        
        guard let phoneSetFrequency = phoneSetFrequency else {
            return
        }
        phoneMotionManager.deviceMotionUpdateInterval = 1.0 / phoneSetFrequency
        
        if phoneMotionManager.isDeviceMotionAvailable {
            //phoneStatusLabel.text = "Phone \(phoneMotionManager.isDeviceMotionAvailable)"
            phoneCnt = 0
            phonePrevTime = NSDate().timeIntervalSince1970
            DispatchQueue.main.async {
                self.phoneStatusLabel.text = "Phone started"
            }
            // phoneMotionManager.startDeviceMotionUpdates(using: .xMagneticNorthZVertical, to: phoneQueue) { (motion, error) in
            phoneMotionManager.startDeviceMotionUpdates(using: .xArbitraryCorrectedZVertical, to: phoneQueue) { (motion, error) in
                if let motion = motion {
                    let currentTime = NSDate().timeIntervalSince1970
                    
                    if let socketClient = self.socketClient {
                        if socketClient.connection.state == .ready {
                            let text = "phone:\(currentTime) \(motion.userAcceleration.x) \(motion.userAcceleration.y) \(motion.userAcceleration.z) \(motion.attitude.quaternion.x) \(motion.attitude.quaternion.y) \(motion.attitude.quaternion.z) \(motion.attitude.quaternion.w) \(motion.gravity.x) \(motion.gravity.y) \(motion.gravity.z) \(motion.attitude.roll) \(motion.attitude.pitch) \(motion.attitude.yaw)\n"
                            socketClient.send(text: text)
                        }
                    }
                    
                    self.phoneCnt += 1
                    if self.phoneCnt % self.nToMeasureFrequency == 0 {
                        let timeDiff = (currentTime - self.phonePrevTime) as Double
                        self.phonePrevTime = currentTime
                        self.phoneMeasuredFrequency = 1.0 / timeDiff * Double(self.nToMeasureFrequency)
                        DispatchQueue.main.async {
                            self.phoneStatusLabel.text = "\(self.phoneCnt) data / \(round(self.phoneMeasuredFrequency! * 100) / 100) [Hz] (set as \(self.phoneSetFrequency!) [Hz])"
                        }
                    }
                } else {
                    print(error as Any)
                }
            }
        }
    }
    
    @IBAction func stopPhoneMotion() {
        
        phoneStatusLabel.text = "not recording"
        phoneMotionManager.stopDeviceMotionUpdates()
    }
}

extension UIViewController {

    @objc func hideKeyboardWhenTapped() {
        let tap: UITapGestureRecognizer = UITapGestureRecognizer(target: self, action: #selector(UIViewController.dismissKeyboard))
        tap.cancelsTouchesInView = false
        view.addGestureRecognizer(tap)
    }

    @objc func dismissKeyboard() {
        view.endEditing(true)
    }
}

extension ViewController: UITextFieldDelegate {
    
    func textFieldShouldReturn(_ textField: UITextField) -> Bool {
        textField.resignFirstResponder()
        return true
    }
}

extension ViewController: UIPickerViewDelegate, UIPickerViewDataSource {
    func numberOfComponents(in pickerView: UIPickerView) -> Int {
        return 1
    }
    
    func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int {
        return frequencies.count
    }
    
    func pickerView(_ pickerView: UIPickerView, titleForRow row: Int, forComponent component: Int) -> String? {
        return String(format: "%.1f", frequencies[row])
    }
    
    func pickerView(_ pickerView: UIPickerView, didSelectRow row: Int, inComponent component: Int) {
        switch pickerView.tag {
        case 1: // phone
            phoneSetFrequency = frequencies[row]
        default:
            print()
        }
    }
    
}


extension ViewController: CMHeadphoneMotionManagerDelegate {
    
}

extension ViewController: WCSessionDelegate {
    
    
    func sessionDidBecomeInactive(_ session: WCSession) {
        
    }
    
    func sessionDidDeactivate(_ session: WCSession) {
        
    }
    
    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
        if let e = error {
            print("Completed activation with error: \(e.localizedDescription)")
        } else {
            print("Completed activation!")
        }
    }
    
    func session(_ session: WCSession, didReceive file: WCSessionFile) {
        
    }
    
    func session(_ session: WCSession, didReceiveMessage message: [String : Any] = [:]) {
        if let motionData = message["motionData"] as? String {
            // real-time logging on the phone
            
            if let socketClient = self.socketClient {
                if socketClient.connection.state == .ready {
                    let text = "watch:" + motionData
                    socketClient.send(text: text)
                }
            }
        }
    }
}


/*extension ViewController: NISessionDelegate {
 
    func session(_ session: NISession, didUpdate nearbyObjects: [NINearbyObject]) {
        // The session runs with one accessory.
        guard let accessory = nearbyObjects.first else { return }

        if let distance = accessory.distance {
            print(distance)
            
            if let socketClient = self.socketClient {
                if socketClient.connection.state == .ready {
                    socketClient.send(text: "phone_watch_distance: \(distance)")
                }
            }
        }

        if let direction = accessory.direction {
            print(direction)
        }
    }
    
}*/


extension ViewController: CBPeripheralManagerDelegate {
    
    func peripheralManagerDidUpdateState(_ peripheral: CBPeripheralManager) {
        switch peripheral.state {
        case .poweredOn:
            peripheralManager.add(tokenService!)
            peripheralManager.startAdvertising([CBAdvertisementDataServiceUUIDsKey: [tokenServiceUUID]])

        default:
            print("CBManager state is \(peripheral.state)")
            return
        }
    }

    func peripheralManager(_ peripheral: CBPeripheralManager, didReceiveRead request: CBATTRequest) {
        if request.characteristic.uuid.isEqual(iPhoneTokenCharacteristicUUID) {
            if let value = iPhoneTokenCharacteristic?.value {
                if request.offset > value.count {
                    peripheral.respond(to: request, withResult: CBATTError.invalidOffset)
                    print("Read fail: invalid offset")
                    return
                }
                request.value = value.subdata(in: Range(uncheckedBounds: (request.offset, value.count)))
                peripheral.respond(to: request, withResult: CBATTError.success)
            }
        }else {
            print("Read fail: wrong characteristic uuid:", request.characteristic.uuid)
        }
    }
}
