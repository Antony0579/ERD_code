// --------------------------------- //
// reference
// self-defined methods :
//   https://arduino.stackexchange.com/questions/30341/replacing-several-pinmode-and-digitalwrite-pins-with-an-array
// --------------------------------- //

// --------------------------------- //
// 腦波機Trigger控制
// 讀取Serial port的資料(字串'0'-'7')，
// 在triggerPins輸出相對應的2進制輸出。
// --------------------------------- //

//# 跳過VCC、tri[0] => 僅可輸出0 2 4 ... 
//# trigger反轉 => 輸出14 12 10 ... 
// trigger pins
uint8_t triggerPins[3] = {10, 9, 8}; // MSB -> LSB -> LSB2
uint8_t trigger[][3] = {{0, 0, 0},  // 0 -> 241--255
                        {0, 0, 1},  // 1 -> 243--249
                        {0, 1, 0},  // 2 -> 245--250
                        {0, 1, 1},  // 3 -> 247
                        {1, 0, 0},  // 4 -> 249--252
                        {1, 0, 1},  // 5 -> 251
                        {1, 1, 0},  // 6 -> 253
                        {1, 1, 1}}; // 7 -> 255--248
int d = 0;
int state = 999;
int start_t = 0;
byte recv_byte = ' ';

// self-define methods for pinMode and digitalWrite                     
template<size_t size>
void pinMode(const uint8_t (&pin)[size], uint8_t mode) {
    for (size_t i = 0; i < size; ++i) {
        pinMode(pin[i], mode);
    }
}

template<size_t size>
void digitalWrite(const uint8_t (&pin)[size], const uint8_t (&val)[size]) {
    for (size_t i = 0; i < size; ++i) {
        digitalWrite(pin[i], val[i]);
    }
}
                                                               
void setup() {
  pinMode(triggerPins, OUTPUT);
  Serial.begin(115200);
}

// Main loop
void loop() {
  if (Serial.available() > 0) {
    recv_byte = Serial.read();
    Serial.print("Received: ");
    state = int(recv_byte - '0'); // convert ASCII to int 
    Serial.println(state);    
    start_t = millis();
  }

  if (state <= 7) {
    d = millis() - start_t; //傳送資料時間
    if (d < 5) {
      digitalWrite(triggerPins, trigger[state]); 
    }
    else {
      digitalWrite(triggerPins, trigger[0]);
      state = 999;
    }
  }
}
