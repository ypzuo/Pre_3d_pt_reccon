from importer import *

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir_imgs', type=str, required=True, 
	help='Path to shapenet rendered images')
parser.add_argument('--data_dir_pcl', type=str, required=True, 
	help='Path to shapenet pointclouds')
parser.add_argument('--mode', type=str, required=True, 
	help='Latent Matching setup. Choose from [lm, plm]')
parser.add_argument('--exp', type=str, required=True, 
	help='Name of Experiment')
parser.add_argument('--gpu', type=str, required=True, 
	help='GPU to use')
parser.add_argument('--ae_logs', type=str, required=True, 
	help='Location of pretrained auto-encoder snapshot')
parser.add_argument('--category', type=str, required=True, 
	help='Category to train on : \
		["all", "airplane", "bench", "cabinet", "car", "chair", "lamp", \
		"monitor", "rifle", "sofa", "speaker", "table", "telephone", "vessel"]')
parser.add_argument('--bottleneck', type=int, required=True, default=128, 
	help='latent space size')
parser.add_argument('--loss', type=str, required=True, 
	help='Loss to optimize on l1/l2/chamfer')
parser.add_argument('--batch_size', type=int, default=32, 
	help='Batch Size during training')
parser.add_argument('--lr', type=float, default=0.00005, 
	help='Learning Rate')
parser.add_argument('--bn_decoder', action='store_true', 
	help='Supply this parameter if you want bn_decoder, otherwise ignore')
parser.add_argument('--load_best_ae', action='store_true', 
	help='supply this parameter to load best model from the auto-encoder')
parser.add_argument('--max_epoch', type=int, default=30, 
	help='max num of epoch')
parser.add_argument('--print_n', type=int, default=100, 
	help='print_n')
parser.add_argument('--sanity_check', action='store_true', 
	help='supply this parameter to visualize autoencoder reconstructions')

FLAGS = parser.parse_args()

print '-='*50
print FLAGS
print '-='*50

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

BATCH_SIZE = FLAGS.batch_size   	# Training Batch Size
VAL_BATCH_SIZE = FLAGS.batch_size   # Validation Batch Size
NUM_POINTS = 2048					# Number of points predicted
HEIGHT = 128 						# Height of input RGB image
WIDTH = 128 						# Width of input RGB image


def fetch_batch(models, indices, batch_num, batch_size):
	'''
	Input:
		models: list of paths to shapenet models
		indices: list of ind pairs, where 
			ind[0] : model index (range--> [0, len(models)-1])
			ind[1] : view index (range--> [0, NUM_VIEWS-1])
		batch_num: batch_num during epoch
		batch_size: batch size for training or validation
	Returns:
		batch_ip: input RGB image of shape (B, HEIGHT, WIDTH, 3)
		batch_gt: gt point cloud of shape (B, NUM_POINTS, 3)
	Description:
		Batch Loader
	'''

	batch_ip = []
	batch_gt = []

	for ind in indices[batch_num*batch_size:batch_num*batch_size+batch_size]:
		model_path = models[ind[0]]
		img_path = join(FLAGS.data_dir_imgs, model_path, 'rendering', PNG_FILES[ind[1]])
		pcl_path = join(FLAGS.data_dir_pcl, model_path, 'pointcloud_2048.npy')

		pcl_gt = np.load(pcl_path)

		ip_image = cv2.imread(img_path)[4:-5, 4:-5, :3]
		ip_image = cv2.cvtColor(ip_image, cv2.COLOR_BGR2RGB)
		batch_gt.append(pcl_gt)
		batch_ip.append(ip_image)

	batch_gt = np.array(batch_gt)
	batch_ip = np.array(batch_ip)
	return batch_ip, batch_gt, ip_image








if __name__ == '__main__':


    # Get Training Models
    train_models, val_models, train_pair_indices, val_pair_indices = get_shapenet_models(FLAGS)
    batches = len(train_pair_indices) / BATCH_SIZE
    for b in xrange(batches):
        batch_ip, batch_gt, ip_image = fetch_batch(train_models, train_pair_indices, b, BATCH_SIZE)
        cv2.imshow('img', ip_image)
        cv2.waitKey(0)