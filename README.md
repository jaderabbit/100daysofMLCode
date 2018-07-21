# 100 Days of ML Code
My log for #100daysofMLCode

## Day 0: 9th July 2018

- Decided on my first project: To program and train a GAN to generate album covers, using Keras
- This is to (a) Learn how GANs work (b) So I can teach others how to use a GAN to do cool stuff with them (got a presentation end of this month I'd like to have this ready for)
- Found & downloaded myself a [nice dataset](https://blog.archive.org/2015/05/27/experiment-with-one-million-album-covers/)
- Found a nice example of someone else who have used a DCGAN to generate album covers: https://github.com/Newmu/dcgan_code
- Found an example of the [DCGAN in Keras](https://github.com/eriklindernoren/Keras-GAN#dcgan)

## Day 1: 10th July 2018

- Started working through this iconic Keras blog: [Building Powerful Image Classification Models Using Very Little Data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html). I discovered how cool the Keras image generating functions are. I was worried about having to annoyingly rescale things. The ImageGenerator is handling that with PIL under the hood. So great
- I've coded the import on my data
- I went on a side trip to discover what sized images GANs perform well on and came across this little wondrous summary: [Fantastic GANs and Where to Find Them](http://guimperarnau.com/blog/2017/03/Fantastic-GANs-and-where-to-find-them). Describes each of the GANs and when you should think about using them. Looks like I should be using an Improved DCGAN or an Improved Wasserstein GAN. 
- Going with image scaling of 128x128 pixels as I've seen examples where Improved DCGAN worked fine on this size
- I started running through the DCGAN in Keras code to understand it in more detail
- Created my GPU instance on Google Cloud running the tensorflow docker container. Docker is bugging out. Will fix tomorrow 


## Day 2: 11th July 2018

- Managed to get my GCloud GPU instance to run the tensorflow docker container, but for some reason can't attach to the docker when I've ssh'd into the box. It just hangs. Been fighting it. Anyone else experienced this?
- Started working through some example Wasserstein GAN code but pretty tired so thats been hard. Tomorrow I'll start with this 
- Found the command. GCloud linux isn't running bash. Weird. Used another command and it worked
- Now cuda is fighting me. Why are you like this?

## Day 3: 14th July 2018

- 12th and 13th got busy so did a couple of hours
- Have abandoned google cloud. Tried out AWS - also needed to request a limit increase for GPU machine. Azure is super expensive
- Paperspace provides gpu powered jupyter notebooks and that works.
- I have bugs in my GAN code. Fighting through them. I predict I'll need a nice deep dive into GAN architectures tomorrow

## Day 4: 15th July 2018

- Got AWS to give me a GPU box. YAAASS. Would prefer that to only having jupyter notebook access
- Realised I jumped in without understanding the Generator architecture (Upsampling, Convolution Transformation layers, etc) so I went back to the basics. Drew many pictures of layers with their dimensions.
- Ended up on a refresher for CNNs. Wanted to make sure I understood the discriminator properly and wasn't feeling great about the core things so read the [Standford CS231 Notes](http://cs231n.github.io/convolutional-networks/)
- Finally ended up trying to understand Batch Normalization by reading some of the original paper [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)


## Day 5: 16th July 2018
- While trying to build a GAN architecture, I realised I had no idea how they got to the numbers of layers, kernels etc, that were being used by the architectures. I _THANKFULLY_ stumbled onto [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285v1) which is a detailed, clear and generally magnificent explanation of how one calculates the sizes of layers, and parameters for the convolutions and the transposed convolutions needed in GANs. I recommend it to everyone.
- I used my learnings to practice calculating the dimensions of layers the architectures for some existing Convnets and GANS. 
- Tomorrow, I will actually code the architecture for my GAN

## Day 6: 17th July 2018
- Managed to make a GAN architecture in Keras that compiles. Yayz! And I used the new Conv2DTranspose layer. Woot!
- Then the problem was that my images had bad EXIF data or other corrupt nonsense, so now to clean that. Thankfully found [this](https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/discussion/31558)
- Then PIL was failing to find certain images. That turned out to be corrupt jpeg files. Wrote a lil bash script to detect invalid images so I can delete them. Thank you ImageMagick for the verify command
```
#!/bin/bash
for filename in *.jpg; do  
  error=`identify -regard-warnings $filename 2>&1 >/dev/null;`
  if [ $? -ne 0 ]; then
    echo "The image is corrupt or unknown format"
    echo "$error"
    echo $filename >> $1
  fi
done
```
And delete with `xargs -d '\n' rm < $1`
- Tomorrow, I train my GAN on a subset of the data and clean the rest of the data

## Day 7: 18th July 2018
- Using my AWS GPU box. I just feel more comfortable running things in a VM. Need to practice using things like Colab or Datalab.
- Currently downloading the larger dataset onto it. 10GB of albums of bands beginning with the word 'The' (Yes, where many of the best bands are). It's going to take 5 hours. I tried torrenting it. Discovered rtorrent (a terminal torrent client), but it gave me a hard time, so back to `wget` it is.
- Did get the GAN training on a much smaller subset of data (All bands beginning with letter `x`). It's taking about 5 seconds per epoch with a batch size of 32. Now to babysit! Let's see if it does anything!

## Day 8: 19th July 2018
- My GAN sorta trained. It was getting somewhere but not far enough. My losses were very large. I'm still working out if that is a good thing. Need to read more into the Weisserstein loss metric. Started on that. It looks like a variation of cosine distance, but more to read!
- Prepped the larger dataset of images (approx. 77000 images) and retrained. Here are some of the images I got:
  ![alt text](https://raw.githubusercontent.com/jaderabbit/100daysofMLCode/master/images/gan.png "GAN1")
  ![alt text](https://raw.githubusercontent.com/jaderabbit/100daysofMLCode/master/images/gan2.png "GAN2")

- I see that the losses start converging to a much smaller range. I'm going to try train it for more epochs and see how that does.
- Started planning my GAN presentation for next week: [Rabbiteers](https://www.meetup.com/Rabbiteer/events/252624998/)
- Started reading the original GAN paper by Ian Goodfellow
- Some goals for tomorrow and Saturday: 
  - Understand out exactly how training works and why it is the way it is
  - Understand exactly how the weisserstein works and why it is better and what are good values for performance
  - Analyse the performance of my GAN trained for longer. See if we're getting any artifacts
  - Modify the code to track more metrics properly

## Day 9: 21st July 2018
- So I trained for 10 times longer (40 000 epochs), but not significant improvement so I'm debugging
- Noticed my images were in a (0,1) input range. I've wrapped the keras generator to scale between (0,1)
- I noticed the image dataset has LOADS of duplicates. Using [fdupes](https://github.com/adrianlopezroche/fdupes) to remove the duplicates.   
- I noticed that there were gifs so removed them. 
- I also wanted to save "progress" generated images so I'm doing that now to see how training goes
- Updated some things in my architecture