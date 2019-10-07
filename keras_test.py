import warnings, logging, os, sys, time, cv2, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
from keras.models import load_model
import numpy as np


default_model_name = '(32,32)LineFollow.h5'
default_dir = os.getcwd().replace('\\','/')
default_img_name = '001.jpg'
model_translator=np.array(['L','N','R','U'])

for r, n, f in os.walk(default_dir):
	break
for i in f:
	if i.endswith('.jpg') or i.endswith('.JPG'):
		default_img_name=i
		break
for i in f:
	if i.endswith('.h5') or i.endswith('.H5'):
		default_model_name=i
		break

def prepare_model(args):
###################################### Initialization Tools ##################################
	model_location = args.model_dir + '/' + args.model_name
	img_location = args.img_dir + '/' + args.img_name
	prepper = np.ones([1, args.res, args.res, 1])
	img = cv2.imread(img_location)

###################################### For Timings #######################################
	if not args.silence:
		t = time.time()
		model = load_model(model_location)
		t = time.time()-t
		print('\nThe model took '+str(t)+' seconds to load.\n')
		
		t = time.time()
		model.predict_classes(prepper)
		t = time.time()-t
		print('\nThe model took '+str(t)+' seconds to initialize.\n')
		
		t = time.time()
		img=cv2.resize(img,(args.res,args.res),interpolation = cv2.INTER_AREA)
		img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = np.expand_dims(img, axis=0)
		img = np.expand_dims(img, axis=-1)
		prediction = model.predict_classes(img/.255)[0]
		t = time.time()-t
		print('\nThe model took '+str(t)+' seconds to generate '+model_translator[prediction]+' as the output for '+args.img_name+'.\n')

###################################### For Silence ########################################
	else:
		model = load_model(model_location)
		model.predict_classes(prepper)
		t = time.time()
		img = cv2.resize(img, (args.res, args.res))
		img = np.expand_dims(img, axis=0)
		img = np.expand_dims(img, axis=-1)
		prediction = model.predict_classes(img)

###################################### For Plotting ########################################
	if args.plot:
		from keras.utils import plot_model
		plot_model(model, to_file=args.model_name.replace('.h5','.png'), show_layer_names=False, show_shapes=True, rankdir='TB')

	return model


def main():
###################################### Parsing The Input Arguments ############################################################
	parser = argparse.ArgumentParser(	description='KerasTest',
																	epilog='EX:\npython keras_test.py --img-dir C:/Users/logan/Desktop/keras_test  --img-name 002.jpg --plot')
	parser.add_argument('--silence', 				action='store_true', 	default=False, 								help='dont show timings')
	parser.add_argument('--plot', 						action='store_true', 	default=False, 								help='make a .png diagram of the model')
	parser.add_argument('--res', 						type=int, 					default=50, 		metavar='S', 		help='resolution ( default: 50 )')
	parser.add_argument('--model-name', 		action='store', 			default=default_model_name, 	help='filename of the model( default: '+default_model_name+' )')
	parser.add_argument('--model-dir', 			action='store', 			default=default_dir, 						help='directory to the model( default: '+default_dir+' )')
	parser.add_argument('--img-name', 			action='store', 			default=default_img_name, 			help='filename of the img( default: '+default_img_name+' )')
	parser.add_argument('--img-dir', 				action='store', 			default=default_dir, 						help='directory to the img( default: '+default_dir+' )')
	args = parser.parse_args()
############################################## Run #######################################################################
	model=prepare_model(args)
	
	
if __name__ == '__main__':
	main()