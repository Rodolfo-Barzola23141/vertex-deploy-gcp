from kfp.compiler import Compiler
from pipelines import train_pipeline, prediction_pipeline  

# Compilamos los pipelines
Compiler().compile(pipeline_func=train_pipeline, package_path='train_pipeline.yaml')
Compiler().compile(pipeline_func=prediction_pipeline, package_path='prediction_pipeline.yaml')

