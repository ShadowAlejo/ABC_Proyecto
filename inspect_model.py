import pickle
import sys

def main():
    try:
        data = pickle.load(open('dataset/models/svm_facial/svm_facial_model.pkl', 'rb'))
        print('class_names:', data['class_names'])
        print('classes:', data['model'].classes_)
    except Exception as e:
        print("Error:", e)

if __name__ == '__main__':
    main()
