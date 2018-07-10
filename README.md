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
