# import sagemaker
# from sagemaker.pytorch import PyTorch


# sagemaker_session = sagemaker.Session()
# role = sagemaker.get_execution_role()


# estimator = PyTorch(entry_point='train.py',
#                     source_dir='./',
#                     role=role,
#                     framework_version='1.2.0',
#                     train_instance_count=1,
#                     train_instance_type='ml.p3.2xlarge',
#                     train_volume_size=120,
#                     hyperparameters={'batch-size': 32}
#                    )

# estimator.fit({
#     'train': 's3://sagemaker-1574825945676/container',
#     'config': 's3://sagemaker-1574825945676/cfg'
# })