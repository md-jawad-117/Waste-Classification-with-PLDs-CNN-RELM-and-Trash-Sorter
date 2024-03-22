#include <Servo.h>
Servo myservo_pan;
Servo myservo_tilt;// create servo object to control a servo
// twelve servo objects can be created on most boards
int p=0;

void setup() {
  // Start serial communication at a baud rate of 9600
  myservo_pan.attach(9);
  myservo_tilt.attach(10);
//  myservo_pan.write(90);  
//  myservo_tilt.write(98);
  Serial.begin(9600);
}

void loop() {
  // Check if data is available to read
  if (Serial.available()) {
    // Read the incoming byte
    int p = Serial.read();

    // Perform actions based on the received value
    if(p==119){
        myservo_pan.write(90);  
        myservo_tilt.write(70);
        delay(100);
        Serial.println(p);
      
    }
    else if(p==115){      
        myservo_pan.write(90); 
        myservo_tilt.write(127);
        delay(100);
        Serial.println(p);
    
    }
   else if(p==97){  
        myservo_pan.write(0);
        myservo_tilt.write(71);
        delay(100);
        Serial.println(p);
   }
        
      else if(p==100){
        myservo_pan.write(0); 
        myservo_tilt.write(127);
        delay(100);
        Serial.println(p);
      }
      else{}
//      default:
//  
//        myservo_pan.write(90);  
//        myservo_tilt.write(98);
//        delay(2000);
//        Serial.println("Invalid option");
//        break;
    }
  }
