import argparse
import os
import random
import sys
import torch
from tqdm import tqdm
from torch.optim import Adam
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

def seed_everything(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	torch.cuda.manual_seed_all(seed)

def parse_args(args=None):
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', type=str)
	parser.add_argument('--model_name', type=str)
	parser.add_argument('--model_path', type=str)
	parser.add_argument('--dim1', type=int)
	parser.add_argument('--dim2', type=int)
	parser.add_argument('--lr', type=float)
	parser.add_argument('--epoch', type=int)


	return parser.parse_args(args)

class MyDataset(Dataset):
	def __init__(self,data_path,label_path):
		super().__init__()
		self.data = self.dataloading(data_path)
		self.label = self.dataloading(label_path)


	def __getitem__(self,idx):
		return self.data[idx],self.label[idx]

	def __len__(self):
		return len(self.data)

	def dataloading(self,path):
		data = torch.load(path)
		return data

class MLPModel(nn.Module):
	def __init__(self,dim1,dim2):
		super(MLPModel, self).__init__()
		self.fc1 = nn.Linear(6*4096,dim1)
		self.fc2 = nn.Linear(dim1, dim2)
		self.fc3 = nn.Linear(dim2, 2)

	def forward(self,x):
		x = torch.flatten(x,1)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.fc3(x)

		return x

class MLPTrain:
	def __init__(self,train_data,test_data,val_data,model,model_path,epoch,lr):
		self.model = model
		self.loss = nn.CrossEntropyLoss()
		self.optim = Adam(self.model.parameters(),lr = lr)
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.train_loader = train_data
		self.test_loader = test_data
		self.val_loader = val_data
		self.model_path = model_path
		self.epoch = epoch

	def evaluate_accuracy(self,test_val):
		test_num = 0
		test_correct = 0
		self.model.eval()
		if test_val == "test":
			data_loader = self.test_loader
		else:
			data_loader = self.val_loader

		with torch.no_grad():
			for b_x, b_y in data_loader:
				b_x = b_x.to(self.device)
				b_y = b_y.to(self.device)
				output = self.model(b_x)
				pred_b_y = output.argmax(dim=1)
				test_num += b_x.size(0)
				test_correct += (pred_b_y == b_y).sum().item()
		return test_correct / test_num

	def model_training(self):
		num_epochs = self.epoch
		self.model = self.model.to(self.device)
		training_history = {"epoch": [], "train_loss": [], "train_accuracy": [],
							"test_accuracy": []}
		optim = self.optim
		for epoch in range(num_epochs):
			train_loss = 0
			train_num = 0
			train_correct = 0
			self.model.train()
			for b_x, b_y in tqdm(self.train_loader):
				b_x = b_x.to(self.device)
				b_y = b_y.to(self.device)
				output = self.model(b_x)
				l = self.loss(output, b_y)
				pred_b_y = output.argmax(dim=1)
				optim.zero_grad()
				l.backward()
				optim.step()
				train_loss += l.item() * b_x.size(0)
				train_num += b_x.size(0)
				train_correct += (pred_b_y == b_y).sum().item()

			train_loss_all = train_loss / train_num
			train_acc = train_correct / train_num
			training_history["train_loss"].append(train_loss_all)
			training_history["train_accuracy"].append(train_acc)
			training_history["epoch"].append(epoch)

			val_acc = self.evaluate_accuracy("val")
			training_history["test_accuracy"].append(val_acc)

			print('epoch %d, train_loss: %.4f, train_accuracy: %.4f, test_accuracy: %.4f' %
				  (epoch, train_loss_all, train_acc, val_acc))

		test_acc = self.evaluate_accuracy("test")
		print('test_accuracy: %.4f' % test_acc)

		torch.save(self.model, self.model_path)


if __name__ == "__main__":
	args = parse_args()
	seed_everything(42)

	if not os.path.exists("clfmodel"):
		os.makedirs("clfmodel")

	train_data_path = "MLP_train_data/QA_train_data_"+args.model_name+".pt"
	train_label_path = "MLP_train_data/QA_train_label_"+args.model_name+".pt"
	test_data_path = "MLP_train_data/QA_test_data_" + args.model_name + ".pt"
	test_label_path = "MLP_train_data/QA_test_label_" + args.model_name + ".pt"
	val_data_path = "MLP_train_data/QA_val_data_" + args.model_name + ".pt"
	val_label_path = "MLP_train_data/QA_val_label_" + args.model_name + ".pt"

	train_data = MyDataset(train_data_path, train_label_path)
	test_data = MyDataset(test_data_path, test_label_path)

	try:
		val_data = MyDataset(val_data_path, val_label_path)
	except:
		test_size = int(0.5 * len(test_data))
		valid_size = len(test_data) - test_size
		test_data, val_data = torch.utils.data.random_split(test_data, [test_size, valid_size])

	train_dl = DataLoader(train_data,batch_size=512,shuffle=True)
	test_dl = DataLoader(test_data,batch_size=512,shuffle=True)
	val_dl = DataLoader(val_data, batch_size=512, shuffle=True)



	model = MLPModel(args.dim1,args.dim2)
	model = model.to(torch.bfloat16)

	smlp = MLPTrain(train_dl,test_dl,val_dl,model,args.model_path,args.epoch,args.lr)

	smlp.model_training()

