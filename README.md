# PBGAN

codes of 4 or 6 layers PBGAN with applications

example:
python main.py --dataset=celeba --input_height=108 --output_height=64 --train --crop --structure=0 --option=5

--dataset: svhn celeba cifar10 Imagenet1 
--train: to train a new model or read checkpoint to do visualization
--structure  ==0 #for original,(default)
             ==1 #for binarized with batch norm,
             ==2 #for binarized without batch norm,
             ==3 #for XNOR with batch norm,(not available for 6 layers)
             ==4 #for XNOR without batch norm(not available for 6 layers)

--option: == 5  #generating sample outputs(default)
          == 6  #svhn feature maps, train,split into several files, each contains 10000 vectors
          == 7  #svhn test feature maps, test ,split into several files, each contains 10000 vectors
          == 8  #cifar10 feature maps, train, split into 5 files
          == 9  #cifar10 feature maps, test, split into 5 files
          == 10 #cifar10 feature maps visualizations,"horse picture"

note: when using celeba or other dataset with picture size not 32x32, 
        use --input_height=xxx and output_height=64 and --crop as the example
if using dataset with 32x32 picture, like svhn, cifar10 Imagenet1
simply run:
python main.py --dataset=cifar10 --train --structure=0 --option=5        
