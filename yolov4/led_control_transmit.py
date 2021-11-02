import serial
from time import sleep
import sys
 
COM_PORT = 'COM5'  # 請自行修改序列埠名稱
BAUD_RATES = 9600
ser = serial.Serial(COM_PORT, BAUD_RATES)
#===============================================================================
def LED_CONTROL(state): 
#try:
#    while True:
        # 接收用戶的輸入值並轉成小寫
        #choice = input('按1開燈、按2關燈、按e關閉程式  ').lower()
 
        if state == "release":
            print('傳送釋放指令')
            ser.write(b'LED_ON.')  # 訊息必須是位元組類型
        #    sleep(0.5)              # 暫停0.5秒，再執行底下接收回應訊息的迴圈
        elif state == "unrelease":
            print('傳送不用釋放指令')
            ser.write(b'LED_OFF.')          
        #    sleep(0.5)
#================================================================================          
        '''elif state == 'e':
            ser.close()
            print('再見！')
            sys.exit()'''
        '''else:
            print('指令錯誤…')'''
 
        '''while ser.in_waiting:
            mcu_feedback = ser.readline().decode()  # 接收回應訊息並解碼
            print('控制板回應：', mcu_feedback)'''
            
#except KeyboardInterrupt:
#    ser.close()
#    print('再見！')