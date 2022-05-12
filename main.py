import sys

from PyQt5 import uic
from PyQt5.QtWidgets import *
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# 붓꽃 데이터 세트를 로딩합니다.
iris = load_iris()
iris_data = iris.data
iris_label = iris.target

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
        self.show()

        self.pushButton.clicked.connect(self.open_sub_window)
    def open_sub_window(self):
        window2 = sub_window()
        window2.show()


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

if __name__=="__main__":
    app = QApplication(sys.argv)
    window = main_window()
    window.show()
    app.exec_()

