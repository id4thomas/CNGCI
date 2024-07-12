import os
import wandb

def wandb_set(run_name: str, config: dict):
	"""
	Add config as artiface
	"""
	run = wandb.init(
		project=os.environ["WANDB_PROJECT"],
		entity=os.environ["WANDB_ENTITY"],
		config=config,
		name=run_name
	)
	# artifact2 = wandb.Artifact('config', type='config')
	# artifact2.add_file(config_dir)
	# run.log_artifact(artifact2)