import sys
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QTextEdit, QLabel, QFileDialog, QComboBox
from PyQt5.QtGui import QPixmap, QFont, QPalette, QBrush, QImage
from PyQt5.QtCore import Qt
import serial
from tensorflow.keras.models import load_model
import shap
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

video_path = "E:/Research_Papers/Nahid sir/Waste_Class_nahid_Sir/DOC/vi1deo.mp4"

images = np.load("E:/Research_Papers/Nahid sir/Waste_Class_nahid_Sir/Dataset/CatTrashX1.npy")
y = np.load("E:/Research_Papers/Nahid sir/Waste_Class_nahid_Sir/Dataset/CatTrashy1.npy")
X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.10, stratify=y, random_state=2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, stratify=y_train, random_state=2)

class_labels_4_class = ['Hazard_Waste', 'Household_Food_waste', 'Recyclable_waste', 'Residual_waste']
class_labels_12_class = ['Battery', 'Expired Food', 'Brown Glass', 'Cardboard','Clothes', 'Green Glass', 'Metal', 'Paper','Plastic', 'Shoes', 'White Glass', 'Trash']


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Set the fixed size for the main window
        self.setFixedSize(1280, 700)

        # Set the background image
        background_image_path = "E:/Research_Papers/Gastro_Nahid_sir_Faisal_vai/Code/background.jpg"
        self.set_background_image(background_image_path)
        self.setWindowTitle("Jawad")
        # Create the main layout
        main_layout = QVBoxLayout(self)

        # Upper horizontal layout with two sub-layouts
        upper_layout = QHBoxLayout()

        # Left layout with buttons and dropdown
        left_layout = QVBoxLayout()

        # Set a name for the APP above the first button
        app_name_label = QLabel('Waste Classification', self)
        app_name_label.setAlignment(Qt.AlignCenter)
        app_name_label.setStyleSheet('font-size: 30pt; font-weight: bold; border: 2px solid black; border-radius: 5px; background-color: #faf3cd; font-family: "Monotype Corsiva";')
        left_layout.addWidget(app_name_label)

        # Dropdown for model selection
        # Dropdown for model selection
        self.model_selector = QComboBox(self)
        self.model_selector.addItems(['Final_Model', 'Final_Model_', 'PLDs-CNN-RELM'])
        
        font = QFont()
        font.setPointSize(14)  # Set the font size to 14 points
        self.model_selector.setFont(font)
        left_layout.addWidget(self.model_selector)

        # Create buttons with object names
        button1 = QPushButton('Classification_4_Class', self, objectName='BlueButton1')
        button2 = QPushButton('Classification_12_Class', self, objectName='BlueButton2')
        button3 = QPushButton('Conveyer Classification', self, objectName='BlueButton1')
        button4 = QPushButton('Clear Text Box', self, objectName='BlueButton2')

        # Decrease button height and set Constantia font
        button_size = (400, 45)  # Adjusted height
        constantia_font = QFont("Constantia", 18)  # Set Constantia font
        button1.setFixedSize(*button_size)
        button1.setFont(constantia_font)
        button1.setStyleSheet('QPushButton#BlueButton1:hover { background-color: #faeea7; color: #000; } QPushButton#BlueButton1:pressed { background-color: #faeea7; } '
                              'border: 2px solid black; border-radius: 15px; background-color: #faeea7;')  # Set border, border-radius, and hover/pressed styles
        button2.setFixedSize(*button_size)
        button2.setFont(constantia_font)
        button2.setStyleSheet('QPushButton#BlueButton2:hover { background-color: #faf3cd; color: #000; } QPushButton#BlueButton2:pressed { background-color: #faf3cd; } '
                              'border: 2px solid black; border-radius: 15px; background-color: #faf3cd;')  # Set border, border-radius, and hover/pressed styles
        button3.setFixedSize(*button_size)
        button3.setFont(constantia_font)
        button3.setStyleSheet('QPushButton#BlueButton1:hover { background-color: #faeea7; color: #000; } QPushButton#BlueButton1:pressed { background-color: #faeea7; } '
                              'border: 2px solid black; border-radius: 15px; background-color: #faeea7;')  # Set border, border-radius, and hover/pressed styles
        button4.setFixedSize(*button_size)
        button4.setFont(constantia_font)
        button4.setStyleSheet('QPushButton#BlueButton2:hover { background-color: #faf3cd; color: #000; } QPushButton#BlueButton2:pressed { background-color: #faf3cd; } '
                              'border: 2px solid black; border-radius: 15px; background-color: #faf3cd;')  # Set border, border-radius, and hover/pressed styles

        # Reduce the gap between buttons
        left_layout.setSpacing(5)

        # Connect buttons to common slots
        button1.clicked.connect(self.button1_clicked)
        button2.clicked.connect(self.button2_clicked)
        button3.clicked.connect(self.button3_clicked)
        button4.clicked.connect(self.button4_clicked)

        # Add buttons to the left layout
        left_layout.addWidget(button1)
        left_layout.addWidget(button2)
        left_layout.addWidget(button3)
        left_layout.addWidget(button4)

        # Right layout with text edit box
        right_layout = QVBoxLayout()

        # Set the font size of the text edit box to 20
        text_edit = QTextEdit(self)
        text_edit.setFontPointSize(18)
        text_edit.setStyleSheet('border: 3px solid black; border-radius: 5px;')  # Set border and border-radius

        # Make the text edit box read-only
        text_edit.setReadOnly(True)

        right_layout.addWidget(text_edit)

        # Add left and right layouts to the upper layout
        upper_layout.addLayout(left_layout)
        upper_layout.addLayout(right_layout)

        # Set a fixed height for the upper layout
        upper_layout_widget = QWidget()
        upper_layout_widget.setLayout(upper_layout)
        upper_layout_widget.setFixedHeight(300)

        # Add the upper layout to the main layout
        main_layout.addWidget(upper_layout_widget)

        # Lower layout for displaying an image
        lower_layout = QVBoxLayout()
        lower_layout.setSizeConstraint(QVBoxLayout.SetFixedSize)  # Set a fixed size for the layout

        # QLabel for displaying the image with black background
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet('border: 3px solid black; background-color: #ffffff; border-radius: 5px;')  # Set border and border-radius

        # Set black background color
        palette = self.image_label.palette()
        palette.setColor(self.image_label.backgroundRole(), Qt.black)
        self.image_label.setAutoFillBackground(True)
        self.image_label.setPalette(palette)

        lower_layout.addWidget(self.image_label)

        # Add the lower layout to the main layout
        main_layout.addLayout(lower_layout)

    def set_background_image(self, image_path):
        # Set background image for the entire app
        palette = QPalette()
        brush = QBrush(QPixmap(image_path))
        palette.setBrush(QPalette.Window, brush)
        self.setPalette(palette)

    def button1_clicked(self):
        
        selected_model_name = self.model_selector.currentText()
        model_path_4_class = f'E:/Research_Papers/Nahid sir/Waste_Class_nahid_Sir/model/4_class/{selected_model_name}.h5'
    
    # Load models
        self.custom_model_4_class = load_model(model_path_4_class)
        # Open a file dialog to select an image
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, 'Open Image', '', 'Image files (*.png *.jpg *.bmp);;All files (*)')

        # Display the selected image in the QLabel
        if file_path:
            IMG_SIZE=124
            image = cv2.imread(file_path)
            # img1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img1 = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            img1 = img1 / 255.0
            img1 = np.expand_dims(img1, axis=0)
            text_edit = self.findChild(QTextEdit)
            text_edit.append(f'=> Selected Model Name: PLDs-CNN-RELM_4_Class')
            text_edit.append(f'')
            
            start_time = time.time()
            predictions = self.custom_model_4_class.predict(img1)
            end_time = time.time()
            elapsed_time1 = end_time - start_time
            top_indices = np.argsort(predictions[0])[::-1][:4]
            top_classes = [class_labels_4_class[i] for i in top_indices]
            top_confidences = [predictions[0][i] for i in top_indices]
            # Print the predicted class and confidence for the top result
            predicted_class_index = np.argmax(predictions)
            predicted_class = class_labels_4_class[predicted_class_index]
            confidence = predictions[0][predicted_class_index]
            rounded_labels = np.argmax(predictions, axis=1)
            print(f"Predicted Class: {predicted_class}, Confidence: {confidence}")
            
            
            if selected_model_name == "Final_Model_":
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(convert_to_Qt_format)

                # Reduce the width while maintaining the aspect ratio
                desired_width = 500  # Set this to your desired width
                resized_pixmap = pixmap.scaledToWidth(desired_width, Qt.SmoothTransformation)

                # Set the resized pixmap to the label
                self.image_label.setPixmap(resized_pixmap)
                # The setScaledContents call is not necessary if you're setting a fixed size
                # self.image_label.setScaledContents(True)  

                
            else:
                shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
                shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough
                background = X_train[np.random.choice(X_train.shape[0], 10, replace=False)]
                e = shap.DeepExplainer(self.custom_model_4_class, background)
                shap_values = e.shap_values(img1)
                
                shap.image_plot(shap_values, img1, show = False)
                plt.savefig("shap_image_app.jpg", bbox_inches="tight", pad_inches=0.2)
                
                pixmap = QPixmap("shap_image_app.jpg")
                self.image_label.setPixmap(pixmap)
                self.image_label.setScaledContents(True)
            
            
            text_edit.append(f'=> Predicted Main Class: {predicted_class}_{rounded_labels}')
            text_edit.append(f'=> Time Taken: {elapsed_time1} seconds')
            text_edit.append(f'')
            for i in range(4):
                text_edit.append(f"Class: {top_classes[i]}, Confidence: {top_confidences[i]*100} %")
            
            
    
    def button3_clicked(self):
        arduino = serial.Serial('COM3', 9600)
        selected_model_name = self.model_selector.currentText()
        model_path_4_class = f'E:/Research_Papers/Nahid sir/Waste_Class_nahid_Sir/model/4_class/{selected_model_name}.h5'
        self.custom_model_4_class = load_model(model_path_4_class)
        cap = cv2.VideoCapture(video_path)
        img_size = 124
        process_frame = True
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if process_frame:
            # Image processing
                img = cv2.resize(frame, (img_size, img_size))
                img = img / 255.0
                img = np.expand_dims(img, axis=0)

                # Model prediction
                start_time = time.time()
                predictions = self.custom_model_4_class.predict(img)
                end_time = time.time()
                elapsed_time2 = end_time - start_time
                rounded_labels = np.argmax(predictions, axis=1)

                # Print or store the results
                print(rounded_labels)
                predicted_class_index = np.argmax(predictions)
                predicted_class = class_labels_4_class[predicted_class_index]
                # Optional: Display the frame (you can also display prediction results on the frame)
                print(predicted_class)
                
                if predicted_class == 'Hazard_Waste':
                    p='s'
                    print(p)
                elif predicted_class == 'Household_Food_waste':
                    p="w"
                elif predicted_class == 'Recyclable_waste':
                    p="a"
                    print(p)
                elif predicted_class == 'Residual_waste':
                    p="d"
                    
                else:
                    continue
            
                # cv2.imshow('Frame', frame)
                
                arduino.write(p.encode())
                # Close the connection
                
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(convert_to_Qt_format)
                # Reduce the width while maintaining the aspect ratio
                desired_width = 700  # Set this to your desired width
                resized_pixmap = pixmap.scaledToWidth(desired_width, Qt.SmoothTransformation)
                # Set the resized pixmap to the label
                self.image_label.setPixmap(resized_pixmap)
                
                top_indices = np.argsort(predictions[0])[::-1][:4]
                top_classes = [class_labels_4_class[i] for i in top_indices]
                top_confidences = [predictions[0][i] for i in top_indices]
                text_edit = self.findChild(QTextEdit)
                text_edit.clear()
                text_edit.append(f'=> Selected Model Name: PLDs-CNN-RELM_4_Class')
                text_edit.append(f'')
                text_edit.append(f'=> Predicted Class: {predicted_class}_{rounded_labels}')
                text_edit.append(f'=> Prediction Time: {elapsed_time2} seconds')
                text_edit.append(f'')
                for i in range(4):
                    text_edit.append(f"Class: {top_classes[i]}, Confidence: {top_confidences[i]*100} %")

                
                
            process_frame = not process_frame
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            # Press Q on keyboard to exit the loop
        cap.release()
        cv2.destroyAllWindows() 
            
            
    def button2_clicked(self):
        selected_model_name = self.model_selector.currentText()
        model_path_12_class = f'E:/Research_Papers/Nahid sir/Waste_Class_nahid_Sir/model/12_class/{selected_model_name}12.h5'
    
    # Load models
        self.custom_model_12_class = load_model(model_path_12_class)
        # Open a file dialog to select an image
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, 'Open Image', '', 'Image files (*.png *.jpg *.bmp);;All files (*)')

        # Display the selected image in the QLabel
        if file_path:
            IMG_SIZE=124
            image = cv2.imread(file_path)
            # img2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img2 = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            img2 = img2 / 255.0
            img2 = np.expand_dims(img2, axis=0)
            text_edit = self.findChild(QTextEdit)
            text_edit.append(f'=> Selected Model Name: PLDs-CNN-RELM_12_Class')
            text_edit.append(f'')
            
            start_time = time.time()
            predictions = self.custom_model_12_class.predict(img2)
            end_time = time.time()
            elapsed_time1 = end_time - start_time
            top_indices = np.argsort(predictions[0])[::-1][:4]
            top_classes = [class_labels_12_class[i] for i in top_indices]
            top_confidences = [predictions[0][i] for i in top_indices]
            # Print the predicted class and confidence for the top result
            predicted_class_index = np.argmax(predictions)
            predicted_class = class_labels_12_class[predicted_class_index]
            confidence = predictions[0][predicted_class_index]
            rounded_labels = np.argmax(predictions, axis=1)
            print(f"Predicted Class: {predicted_class}, Confidence: {confidence}")
            
            if selected_model_name == "Final_Model_":
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(convert_to_Qt_format)

                # Reduce the width while maintaining the aspect ratio
                desired_width = 500  # Set this to your desired width
                resized_pixmap = pixmap.scaledToWidth(desired_width, Qt.SmoothTransformation)

                # Set the resized pixmap to the label
                self.image_label.setPixmap(resized_pixmap)
                # The setScaledContents call is not necessary if you're setting a fixed size
                # self.image_label.setScaledContents(True)  
            
            else:
                shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
                shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough
                background = X_train[np.random.choice(X_train.shape[0], 10, replace=False)]
                e = shap.DeepExplainer(self.custom_model_12_class, background)
                shap_values = e.shap_values(img2)
                shap.image_plot(shap_values, img2, show = False)
                plt.savefig("shap_image_app.jpg", bbox_inches="tight", pad_inches=0.2)
                pixmap = QPixmap('shap_image_app.jpg')
                self.image_label.setPixmap(pixmap)
                self.image_label.setScaledContents(True)
            
            text_edit.append(f'=> Predicted Main Class: {predicted_class}_{rounded_labels}')
            text_edit.append(f'=> Time Taken: {elapsed_time1} seconds')
            text_edit.append(f'')
            for i in range(4):
                text_edit.append(f"Class: {top_classes[i]}, Confidence: {top_confidences[i]*100} %")









    def button4_clicked(self):
        # Clear the text edit box when Button 4 is clicked
        text_edit = self.findChild(QTextEdit)
        text_edit.clear()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())

