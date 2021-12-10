# Fixed-point UNet Micro-Architecture Simulator

## Dependency

- python 3.+

## Environenment Setup

clone repository
```
git clone https://github.com/willytai/209Final.git
```

create virtual environment and install requirements
```
cd 209Final
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

## Inference

### Model

Weights for model `UNet/unet_model.json` can be found here  
- floating-point weight: [unet\_model.hdf5](https://drive.google.com/file/d/1ED_2y6CAPgSV4XP-Ytnb-5D8bijMZITr/view?usp=sharing)

Weights for model `Unet/unet_model_light.json` are uploaded to `UNet/`
- floating-point weights: `UNet/unet_model_light.hdf5`
- 8-bit fixed-point weights: `UNet/unet_model_light_fixed8.hdf5`
- 16-bit fixed-point weights: `UNet/unet_model_light_fixed16.hdf5`

### Run Script

testing data are in `UNet/testData/`  
to run inference loop for the light-weight model with default parameters floating-point operation:
```
python3 Sim.py --model UNet/unet_model_light.json \
               --weight UNet/unet_model_light.hdf5 \
               --input UNet/testData/0.png
```
results will be stored in `Result/` by default  
  
to specify the PE array size and configure the simulator to run with fixed-point arithmetic:
```
python3 Sim.py --model UNet/unet_model_light.json \
               --weight UNet/unet_model_light.hdf5 \
               --input UNet/testData/0.png \
               --pe <Number of PE> \
               --fixed <World Length for the Fixed-Point>
```

run
```
python3 Sim.py -h
```
for more options
