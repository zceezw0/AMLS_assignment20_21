import preprocess
from params import *
from A1.task_A1 import SVM_A1
from A2.task_A2 import CNN_A2
from B1.task_B1 import CNN_B1
from B2.task_B2 import CNN_B2
import os
# Set the GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


#########################
#For Task A1
#########################
train_X, test_X, train_Y, test_Y = preprocess.data_preprocessing_A1(images_dir, celeba_dir, labels_filename)
extra_test_X, extra_test_Y = preprocess.extra_preprocessing_A1(images_test_dir, celeba_test_dir, labels_test_filename)
model_A1 = SVM_A1()                 # Build model object.
acc_A1_train, SVM_A1_clf = model_A1.train(train_X, train_Y, test_X, test_Y) # Train model based on the training set (you should fine-tune your model based on validation set.)
acc_A1_test = model_A1.test(SVM_A1_clf, extra_test_X, extra_test_Y)   # Test model based on the test set.

#########################
#For Task A2
#########################
train_gen, valid_gen, eval_gen, test_gen = preprocess.data_preprocessing_A2(images_dir, os.path.join(celeba_dir, labels_filename), images_test_dir, os.path.join(celeba_test_dir, labels_test_filename))
model_A2 = CNN_A2()
acc_A2_train, model_pathA2 = model_A2.train(A2_dir, 2, train_gen, valid_gen, eval_gen)
acc_A2_test = model_A2.test(model_A2_path, test_gen)

# #########################
# #For Task B1
# #########################
train_genB1, valid_genB1, eval_genB1, test_genB1 = preprocess.data_preprocessing_B1(cartoon_images_dir, labels_path, cartoon_images_test_dir, labels_test_path)
model_B1 = CNN_B1()
acc_B1_train, model_pathB1 = model_B1.train(B1_dir, 5, train_genB1, valid_genB1, eval_genB1)
acc_B1_test = model_B1.test(model_B1_path, test_genB1)

# #########################
# #For Task B2
# #########################
train_genB2, valid_genB2, eval_genB2, test_genB2 = preprocess.data_preprocessing_B2(cartoon_images_dir, labels_path, cartoon_images_test_dir, labels_test_path)
model_B2 = CNN_B2()
acc_B2_train, model_pathB2 = model_B2.train(B2_dir, 5, train_genB2, valid_genB2, eval_genB2)
acc_B2_test = model_B2.test(model_B2_path, test_genB2)

def print_res(task, dct1, dct2):
	print(task + 'train accuracy: ')
	for item, value in dct1.items():
		print('{}: ({})'.format(item, value))

	print(task + 'test accuracy: ')
	for item, value in dct2.items():
		print('{} ({})'.format(item, value))

print_res('Task A1', acc_A1_train, acc_A1_test)
print_res('Task A2', acc_A2_train, acc_A2_test)
print_res('Task B1', acc_B1_train, acc_B1_test)
print_res('Task B2', acc_B2_train, acc_B2_test)
