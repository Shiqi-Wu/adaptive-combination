This experiment is conducted using reaction-diffusion data model. The hyper parameters are as 
```
params = Reaction_diffusion_param(1)
NN_params = NN_param(1, 32, 10)
model = SimpleModel(params, NN_params)
```
We first train with lambda1 = 1, lambda2 = 0
then train with lambda1 = 1e3, lambda2 = 1