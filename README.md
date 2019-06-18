# Text to Facial Composite
My attempt to train a GAN to generate the facial composites from the description(text or audion input)

## Facial Composite Synthesis using Deep Learning
Facial composites are the sketches drawn by the artist based on the verbal descrip-
tions provided by a person. The reader must have seen this process in popular movies,
where the a suspect's sketch is drawn based on the individual features expressed by
a witness. Hence, the name facial composite, it is a composite of different features.

### Text Encoder
Text Encoder converts the textual description into something the computer or the
model can understand better, presumably some integer representation for each and
every word. For this we require a unique one to one mapping from a word to a
number. One idea is to use a large array of numbers mapped to each and every
unique occurrence of a word in the data set. But, this is a major overhead and time
consuming. It would also be aected by the quality of the data. We must try to make
this process as independent of the data set as possible. For this, we use InferSent by
Facebook for text encoding.

### Conditioning Augmentor
Conditioning Augmentation block (a single linear layer) to obtain the textual part of
the latent vector (uses VAE like reparameterization technique) for the GAN as input.
The second part of the latent vector is random Gaussian noise. The latent vector so
produced is fed to the generator part of the GAN, while the embedding is fed to the
nal layer of the discriminator for conditional distribution matching. The training
of the GAN progresses exactly as mentioned in the ProGAN paper,i.e, layer by
layer at increasing spatial resolutions.

### Progressive GAN architexture
Typically, a GAN consists of two networks: generator and discriminator (aka critic).
The generator produces a sample, e.g., an image, from a latent code, and the distribu-
tion of these images should ideally be indistinguishable from the training distribution.
Since it is generally infeasible to engineer a function that tells whether that is the
case, a discriminator network is trained to do the assessment, and since networks
are dierentiable, we also get a gradient we can use to steer both networks to the
right direction. Typically, the generator is of main interest the discriminator is an
adaptive loss function that gets discarded once the generator has been trained.

The generation of high-resolution images is dicult because higher resolution
makes it easier to tell the generated images apart from training images, thus drastically amplifying the gradient problem. Large resolutions also necessitate using smaller
mini batches due to memory constraints, further compromising training stability. The
key insight is that we can grow both the generator and discriminator progressively,
starting from easier low-resolution images, and add new layers that introduce higher-
resolution details as the training progresses. This greatly speeds up
training and improves stability in high resolutions.

## Flask Web Interface
The project is implemented in python programming language, but a normal user does
not wish to know the technical aspects of using this software to generate the desired
facial composites. Hence, it is noted that a Web Interface can bridge the gap between
the theoretical angle and the output.
