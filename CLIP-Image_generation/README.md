## Image generation and Style Transfer using CLIP embeddings

Inspired from Gradient attack methods, generation/style transfer of images using similarity loss between image and textual embeddings by adding pertubations in multiple steps.


### Brief idea 

**Main Goal**: Minimize Losses

*Similarity Loss*: Difference between cosine similarities of pertubed image and text embeddings.

*Content Loss*: Difference between cosine similarities of pertubed image and original image.

*Total Variance Loss*: Variance /smoothness of image

### General Guidelines
- Start with a content image even for generation and keep content weight = 0 (neeeded to maintain structure)
- Do multiple iterations and vary the parameters.
