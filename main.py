import sys

import matplotlib
from PyQt5 import uic
from PyQt5.QtWidgets import *
from pandas import DataFrame
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

cols = ["sepal len (cm)", "sepal width (cm)", "petal len (cm)", "petal width (cm)"]

# 붓꽃 데이터 세트를 로딩합니다.
iris = load_iris()
iris_data = iris.data
iris_label = iris.target
last_model = "DTC"
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

res_data = []


class test_window(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("test.ui", self)
        self.verticalSlider.valueChanged.connect(self.slider_changed)
        self.train.clicked.connect(self.train_pressed)
        self.knn.clicked.connect(lambda x: self.stateChange("knn"))
        self.dtc.clicked.connect(lambda x: self.stateChange("dtc"))
        self.lr.clicked.connect(lambda x: self.stateChange("lr"))
        self.mlp.clicked.connect(lambda x: self.stateChange("mlp"))
        self.br.clicked.connect(lambda x: self.stateChange("br"))
        self.slider_changed()

        self.xBox.addItems(cols)
        self.yBox.addItems(cols)

        self.xBox.currentTextChanged.connect(lambda x: self.setup())
        self.yBox.currentTextChanged.connect(lambda x: self.setup())

        self.fig = Figure(figsize=(5, 3))
        self.canvas = FigureCanvasQTAgg(self.fig)

        lay = QVBoxLayout()
        lay.addWidget(self.canvas)

        self.widget.setLayout(lay)
        self.setup()

    def setup(self):
        self.fig.clf()
        self.ax = self.fig.subplots()
        self.ax.set_aspect('auto')
        self.ax.grid(True, linestyle='-', color='0.10')

        self.ax.relim()
        self.ax.autoscale_view()

        self.dataFrame = DataFrame(iris_data, columns=cols)
        if len(res_data) == 0:
            self.ax.set_title('Scatter Graph (Default)')
            self.scat = self.ax.scatter(self.dataFrame[self.xBox.currentText()],
                                        self.dataFrame[self.yBox.currentText()], c=iris_label)
        else:
            self.ax.set_title('Scatter Graph (' + last_model + ')')
            self.scat = self.ax.scatter(self.dataFrame[self.xBox.currentText()],
                                        self.dataFrame[self.yBox.currentText()], c=res_data)
        self.scat.set_alpha(0.8)
        self.ax.set_xlabel(self.xBox.currentText())
        self.ax.set_ylabel(self.yBox.currentText())
        self.canvas.draw()

    def stateChange(self, stre):
        self.knn.setChecked(stre == "knn")
        self.dtc.setChecked(stre == "dtc")
        self.lr.setChecked(stre == "lr")
        self.mlp.setChecked(stre == "mlp")
        self.br.setChecked(stre == "br")

    def slider_changed(self):
        self.labelt.setText("훈련에 사용할 데이터: %d개" % self.verticalSlider.value())
        self.labelte.setText("테스트에 사용할 데이터: %d개" % (150 - self.verticalSlider.value()))

    def train_pressed(self):
        global window, last_model, dt_clf
        window.pushButton.setEnabled(True)

        X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label,
                                                            test_size=((150 - self.verticalSlider.value()) / 150),
                                                            random_state=0)
        if self.knn.isChecked():
            last_model = "KNN"
            self.targetMod.setText("마지막으로 KNN 모델로 훈련되었습니다.")
            dt_clf = KNeighborsClassifier(n_neighbors=1)
        elif self.dtc.isChecked():
            last_model = "DTC"
            self.targetMod.setText("마지막으로 DTC 모델로 훈련되었습니다.")
            dt_clf = LogisticRegression()
        elif self.mlp.isChecked():
            last_model = "MLP"
            self.targetMod.setText("마지막으로 MLP 모델로 훈련되었습니다.")
            dt_clf = MLPClassifier()
        elif self.br.isChecked():
            last_model = "BR"
            self.targetMod.setText("마지막으로 BR 모델로 훈련되었습니다.")
            dt_clf = BayesianRidge()
        else:
            last_model = "LR"
            self.targetMod.setText("마지막으로 LR 모델로 훈련되었습니다.")
            dt_clf = DecisionTreeClassifier(random_state=11)
        dt_clf.fit(X_train, y_train)
        prd = dt_clf.predict(X_test)
        scr = accuracy_score(y_test, prd) * 100
        global res_data
        res_data = dt_clf.predict(iris_data)
        self.setup()
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
