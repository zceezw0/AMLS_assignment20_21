import params as l2
import pandas as pd
from params import *
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

def data_preprocessing_A1(images_dir, celeba_dir, labels_filename):
	
	face, genders, _ = l2.extract_features_labels(images_dir, celeba_dir, labels_filename)
	train_X, test_X, train_Y, test_Y = get_data(face, genders)

	return train_X, test_X, train_Y, test_Y

def extra_preprocessing_A1(images_test_dir, celeba_test_dir, labels_test_filename):
	# Extract feature data
	face, genders, _ = l2.extract_features_labels(images_test_dir, celeba_test_dir, labels_test_filename)
	test_X, test_Y = get_test_data(face, genders)

	return test_X, test_Y

def data_preprocessing_A2(images_dir, labels_path):
	# Converting csv into dataframe using read_csv(label_path)
	cartoon_df = pd.read_csv(os.path.normcase(labels_path), sep='\t', engine='python')
	df = cartoon_df[['img_name', 'smiling']]

	# Convert the face shape column to class 1-5
	df.loc[:,'smiling'] += 1

	# Convert face shape column type from int64 to str for data generator
	df = df.applymap(str)
	

	# Setup data generator for train and validation dataset generator
	train_df, test_df = train_test_split(df, train_size=0.7, random_state=42)
	train_datagen = ImageDataGenerator(rescale=1./255,
									   horizontal_flip=True,
									   vertical_flip=True,
									   validation_split=0.3) 

	# Generating training and validation dataset for VGGNet CNN
	train_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
														directory=images_dir,
														x_col="img_name",
														y_col="smiling",
														target_size=(96, 96),
														batch_size=32,
														shuffle=True,
														class_mode='categorical',
														subset='training')
	valid_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
														directory=images_dir,
														x_col="img_name",
														y_col="smiling",
														target_size=(96, 96),
														batch_size=32,
														shuffle=True, 
														class_mode='categorical',
														subset='validation')

	# Evaluate the model with validation dataset
	eval_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
														directory=images_dir,
														x_col="img_name",
														y_col="smiling",
														target_size=(96, 96),
														batch_size=1,
														shuffle=True, 
														class_mode='categorical',
														subset='validation')

	# Generate test dataset from dataframe
	test_datagen = ImageDataGenerator(rescale=1./255)
	test_generator = test_datagen.flow_from_dataframe(dataframe=test_df,
														directory=images_dir,
														x_col="img_name",
														y_col="smiling",
														target_size=(96, 96),
														batch_size=1,
														shuffle=False,
														class_mode='categorical')

	return train_generator, valid_generator, eval_generator, test_generator
def data_preprocessing_B1(images_dir, labels_path):
	# Converting csv into dataframe using read_csv(label_path)
	cartoon_df = pd.read_csv(os.path.normcase(labels_path), sep='\t', engine='python')
	df = cartoon_df[['file_name', 'face_shape']]

	# Convert the face shape column to class 1-5
	df.loc[:,'face_shape'] += 1

	# Convert face shape column type from int64 to str for data generator
	df = df.applymap(str)
	
	# Setup data generator for train and validation dataset generator
	train_df, test_df = train_test_split(df, train_size=0.7, random_state=42)
	train_datagen = ImageDataGenerator(rescale=1./255,
										horizontal_flip=True,
										vertical_flip=True,
										validation_split=0.3) 

	# Generating training and validation dataset for VGGNet CNN
	train_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
														directory=images_dir,
														x_col="file_name",
														y_col="face_shape",
														target_size=(96, 96),
														batch_size=32,
														shuffle=True,
														class_mode='categorical',
														subset='training')
	valid_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
														directory=images_dir,
														x_col="file_name",
														y_col="face_shape",
														target_size=(96, 96),
														batch_size=32,
														shuffle=True, 
														class_mode='categorical',
														subset='validation')

	# Evaluate the model with validation dataset
	eval_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
														directory=images_dir,
														x_col="file_name",
														y_col="face_shape",
														target_size=(96, 96),
														batch_size=1,
														shuffle=True, 
														class_mode='categorical',
														subset='validation')

	# Generate test dataset from dataframe
	test_datagen = ImageDataGenerator(rescale=1./255)
	test_generator = test_datagen.flow_from_dataframe(dataframe=test_df,
														directory=images_dir,
														x_col="file_name",
														y_col="face_shape",
														target_size=(96, 96),
														batch_size=1,
														shuffle=False,
														class_mode='categorical')

	return train_generator, valid_generator, eval_generator, test_generator

def data_preprocessing_B2(images_dir, labels_path):
	# Converting csv into dataframe using read_csv(label_path)
	cartoon_df = pd.read_csv(os.path.normcase(labels_path), sep='\t', engine='python')
	df = cartoon_df[['file_name', 'eye_color']]

	# Convert the face shape column to class 1-5
	df.loc[:,'eye_color'] += 1

	# Convert face shape column type from int64 to str for data generator
	df = df.applymap(str)
	

	# Setup data generator for train and validation dataset generator
	train_df, test_df = train_test_split(df, train_size=0.7, random_state=42)
	train_datagen = ImageDataGenerator(rescale=1./255,
	                                   horizontal_flip=True,
	                                   vertical_flip=True,
	                                   validation_split=0.3) 

	# Generating training and validation dataset for VGGNet CNN
	train_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
	                                                    directory=images_dir,
	                                                    x_col="file_name",
	                                                    y_col="eye_color",
	                                                    target_size=(224, 224),
	                                                    batch_size=32,
	                                                    shuffle=True,
	                                                    class_mode='categorical',
	                                                    subset='training')
	valid_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
	                                                    directory=images_dir,
	                                                    x_col="file_name",
	                                                    y_col="eye_color",
	                                                    target_size=(224, 224),
	                                                    batch_size=32,
	                                                    shuffle=True, 
	                                                    class_mode='categorical',
	                                                    subset='validation')

	# Evaluate the model with validation dataset
	eval_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
	                                                    directory=images_dir,
	                                                    x_col="file_name",
	                                                    y_col="eye_color",
	                                                    target_size=(224,224),
	                                                    batch_size=1,
	                                                    shuffle=True, 
	                                                    class_mode='categorical',
	                                                    subset='validation')

	# Generate test dataset from dataframe
	test_datagen = ImageDataGenerator(rescale=1./255)
	test_generator = test_datagen.flow_from_dataframe(dataframe=test_df,
	                                                    directory=images_dir,
	                                                    x_col="file_name",
	                                                    y_col="eye_color",
	                                                    target_size=(224, 224),
	                                                    batch_size=1,
	                                                    shuffle=False,
	                                                    class_mode='categorical')

	return train_generator, valid_generator, eval_generator, test_generator

def get_data(X, Y):
	Y = np.array([Y, -(Y - 1)]).T
	X, Y = shuffle(X, Y)
	tr_X, te_X, tr_Y, te_Y = train_test_split(X, Y, train_size = 0.7, random_state = 42)
	
	# Reshape training and test X into (n_samples, n_features)
	tr_X = tr_X.reshape(tr_X.shape[0], tr_X.shape[1]*tr_X.shape[2])
	te_X = te_X.reshape(te_X.shape[0], te_X.shape[1]*te_X.shape[2])
	
	# Unzipped training and test Y from (n_samples,) into (n_samples)
	tr_Y = list(zip(*tr_Y))[0]
	te_Y = list(zip(*te_Y))[0]
	
	return tr_X, te_X, tr_Y, te_Y

def get_test_data(X, Y):
	Y = np.array([Y, -(Y - 1)]).T
	
	# Reshape training and test X into (n_samples, n_features)
	te_X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
	
	# Unzipped training and test Y from (n_samples,) into (n_samples)
	te_Y = list(zip(*Y))[0]
	
	return te_X, te_Y