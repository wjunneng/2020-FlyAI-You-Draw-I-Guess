from flyai.dataset import Dataset
from CNN.model import Model

data = Dataset()
model = Model(data)
x_test = [{'json_path': 'draws/draw_490253.json'}, {'json_path': 'draws/draw_56397.json'}]
p = model.predict_all(x_test)
print(p)