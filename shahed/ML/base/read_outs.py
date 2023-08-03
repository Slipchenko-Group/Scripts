import os
import pandas as pd

def read_outs():
	fls = [fl for fl in os.listdir(os.getcwd()) if fl.endswith('.out') and 'slurm' not in fl]
	
	dfs = []
	
	for fl in fls:
		df = pd.read_csv(fl, header=None)
		df = df.rename(columns={0:'epoch',1: 'train_rmse', 2:'test_rmse'})
		df['name'] = fl.replace('.out', '')
		dfs.append(df)
	
	master = pd.concat(dfs)
	master['overfit'] = master['test_rmse'] - master['train_rmse']
	return(master)

master = read_outs()
print(master[abs(master['overfit'])<0.5].sort_values('test_rmse', ascending=True).head(5))

