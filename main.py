import sys

from PyQt5 import uic
from PyQt5.QtWidgets import *
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# 붓꽃 데이터 세트를 로딩합니다.
iris = load_iris()
iris_data = iris.data
iris_label = iris.target

print(len(iris_label))
# DecisionTreeClassifier 객체 생성
dt_clf = DecisionTreeClassifier(random_state=11)

# 학습 수행
print("학습 수행중..")
dt_clf.fit(iris_data, iris_label)

def res_to_name(res):
    if res == 1:
        return "세토사"
    elif res == 2:
        return "버지컬러"
    else:
        return "버지니아"

class main_window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("untitled.ui", self)

        self.pushButton.clicked.connect(self.open_sub_window)
        self.startTrain.clicked.connect(self.open_test_window)
    def open_sub_window(self):
        window2 = sub_window()
        window2.show()
    def open_test_window(self):
        window2 = test_window()
        window2.show()
app = QApplication(sys.argv)
window = main_window()
class test_window(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("test.ui", self)
        self.verticalSlider.valueChanged.connect(self.slider_changed)
        self.train.clicked.connect(self.train_pressed)
        self.knn.clicked.connect(lambda x : self.stateChange("knn"))
        self.dtc.clicked.connect(lambda x: self.stateChange("dtc"))
        self.lr.clicked.connect(lambda x: self.stateChange("lr"))
        self.slider_changed()
    def stateChange(self, stre):
        print(stre)
        if stre == "knn":
            self.knn.setChecked(True)
            self.dtc.setChecked(False)
            self.lr.setChecked(False)
        elif stre == "dtc":
            self.knn.setChecked(False)
            self.dtc.setChecked(True)
            self.lr.setChecked(False)
        else:
            self.knn.setChecked(False)
            self.dtc.setChecked(False)
            self.lr.setChecked(True)
    def slider_changed(self):
        self.labelt.setText("훈련에 사용할 데이터: %d개" % self.verticalSlider.value())
        self.labelte.setText("테스트에 사용할 데이터: %d개" % (150 - self.verticalSlider.value()))
    def train_pressed(self):
        global dt_clf
        global window
        window.pushButton.setEnabled(True)

        X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size=((150 - self.verticalSlider.value()) / 150), random_state=0)
        if self.knn.isChecked():
            self.targetMod.setText("마지막으로 KNN 모델로 훈련되었습니다.")
            dt_clf = KNeighborsClassifier(n_neighbors=1)
        elif self.dtc.isChecked():
            self.targetMod.setText("마지막으로 DTC 모델로 훈련되었습니다.")
            dt_clf = LogisticRegression()
        else:
            self.targetMod.setText("마지막으로 LR 모델로 훈련되었습니다.")
            dt_clf = DecisionTreeClassifier(random_state=11)
        dt_clf.fit(X_train, y_train)
        prd = dt_clf.predict(X_test)
        scr = accuracy_score(y_test, prd) * 100
        self.accLab.setText("정확도: %.2f%%" % scr)


class sub_window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("sub.ui", self)
        self.show()

        self.sliderA.valueChanged.connect(self.slider_changed)
        self.sliderB.valueChanged.connect(self.slider_changed)
        self.sliderC.valueChanged.connect(self.slider_changed)
        self.sliderD.valueChanged.connect(self.slider_changed)

        self.start()

    def update_res(self, res):
        print(res)
        #self.ResultLabel.text = res
    def slider_changed(self):
        self.start()

    def start(self):
        a = self.sliderA.value() / 10
        b = self.sliderB.value() / 10
        c = self.sliderC.value() / 10
        d = self.sliderD.value() / 10
        self.labelA.setText(str("%.2f cm" % a))
        self.labelB.setText(str("%.2f cm" % b))
        self.labelC.setText(str("%.2f cm" % c))
        self.labelD.setText(str("%.2f cm" % d))
        pred_res = dt_clf.predict([[a, b, c, d]])
        self.ResultLabel.setText("이 꽃은 " + res_to_name(pred_res[0]) + " 입니다")

window.show()
app.exec_()

