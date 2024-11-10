#include <SoftwareSerial.h>   // 引用程式庫
#include <Stepper.h> 

// #define MoPin 12
// #define MoPin2 13
// 定義連接藍牙模組的序列埠

SoftwareSerial BT(2, 3); // 接收腳, 傳送腳// RX, TX 反
String str;

//馬達
// setSpeed(long whatSpeed)
// step_delay = 5.85 / 2048* step
// step = step_delay/350
const int stepsPerRevolution = 2048;  
Stepper myStepper1(stepsPerRevolution, 6, 8, 7, 9);
Stepper myStepper2(stepsPerRevolution, 10, 12, 11, 13);

char buff[20];
void setup() {
  Serial.begin(9600);   // 與電腦序列埠連線

  int B = 9600;
  BT.begin(B);
  sprintf(buff,"BT is ready! Baud: %u",B);  
  Serial.println(buff);

  myStepper1.setSpeed(3);//rpm 6/60 = 0.1/s
  myStepper2.setSpeed(3);
}

void loop() {
   
// 若收到「序列埠監控視窗」的資料，則送到藍牙模組
  if (BT.available()) {

    Serial.println("OKOK");
    // Serial.println(BT.read());

    // 讀取傳入的字串直到"\n"結尾
    str = BT.readStringUntil('\n');
    Serial.println(str);

    if (str == "VIB_L") {           // 若字串值是 "LED_ON" 開燈
        myStepper1.step(102);//0.1*2048     // 開燈
        Serial.println("VIB_left is ON"); // 回應訊息給電腦
    } 
    if (str == "VIB_R") {           // 若字串值是 "LED_ON" 開燈
        myStepper2.step(102);//0.1*2048     // 開燈
        Serial.println("VIB_left is ON"); // 回應訊息給電腦
    }
    
  }
}
