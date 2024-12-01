import annoy
from anndata import AnnData
import logging
logger = logging.getLogger(__name__)

class ApproxNN():
	def __init__(self, data):
		self.dimension = data.shape[1]
		self.data = data.astype('float32')

	def build(self, number_of_trees=50):
		self.index = annoy.AnnoyIndex(self.dimension,'angular')
		for i, vec in enumerate(self.data):
			self.index.add_item(i, vec.tolist()[0])
		self.index.build(number_of_trees)

	def query(self, vector, k):
		indexes, distances = self.index.get_nns_by_vector(vector.tolist()[0], k, include_distances=True)
		return list(zip(indexes, distances))


def get_NNmodel(mtx,number_of_trees,use_pca=False):
	
	if use_pca:
		from sklearn.decomposition import PCA
		pca = PCA(n_components=mtx.shape[1])
		mtx2 = pca.fit_transform(mtx) 
		model_ann = ApproxNN(mtx2)
		model_ann.build(number_of_trees)
		return model_ann
	else:
		model_ann = ApproxNN(mtx)
		model_ann.build(number_of_trees)
		return model_ann
	 

def get_neighbours(mtx,model,nbrsize=1):
		
	nbr_dict={}
	for idx,row in enumerate(mtx):
		nbr_dict[idx] = model.query(row,k=nbrsize)[0]
	return nbr_dict

def generate_neighbours(source_adata: AnnData, 
						target_adata: AnnData,
						tag: str,
						number_of_trees:int = 50 
						) -> dict:
	ann_model = get_NNmodel(source_adata.X.todense(),number_of_trees)
	nbr_dict = get_neighbours(target_adata.X.todense(),ann_model)
	return nbr_dict

