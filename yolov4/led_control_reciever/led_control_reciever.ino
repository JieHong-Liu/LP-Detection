#define LED 13
String str;
 
void setup() {
  pinMode(LED, OUTPUT);
  Serial.begin(9600);
  digitalWrite(LED,LOW);
}
 
void loop() {
  if (Serial.available()) {
    // 讀取傳入的字串直到'.'結尾
    str = Serial.readStringUntil('.');
 
    if (str == "LED_ON") {           // 若字串值是 "LED_ON" 開燈
        digitalWrite(LED, HIGH);     // LED一個脈衝
        delay(1000);                 //1 sec
        digitalWrite(LED,LOW);
        Serial.println("Release"); // 回應訊息給電腦

    } else if (str == "LED_OFF") {
        digitalWrite(LED, LOW);
        Serial.println("Unrelease");
    }
  }
}