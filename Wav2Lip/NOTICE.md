# Third-Party Notice: Wav2Lip

The code in this `Wav2Lip/` directory is vendored from the original
[Rudrabha/Wav2Lip](https://github.com/Rudrabha/Wav2Lip) repository
("A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild",
ACM Multimedia 2020). It is **not** covered by this project's top-level MIT
`LICENSE` - it remains subject to the original authors' terms below.

## License restriction (from the upstream repository)

> This repository can only be used for personal/research/non-commercial
> purposes. As the models are trained on the LRS2 dataset, any form of
> commercial use is strictly prohibited. For commercial requests please
> contact the authors directly at rudrabha@synclabs.so or prajwal@synclabs.so.

The upstream project does not ship a separate `LICENSE` file; the restriction
above is the license text as published in their README. If you plan to use
this pipeline (or the `wav2lip.pth` / `s3fd.pth` model weights) for anything
beyond personal, research, or non-commercial use, contact the Wav2Lip authors
first, or use their hosted commercial API at https://sync.so.

## Citation

If you use this pipeline in research, please cite the original paper:

```bibtex
@inproceedings{10.1145/3394171.3413532,
    author = {Prajwal, K R and Mukhopadhyay, Rudrabha and Namboodiri, Vinay P. and Jawahar, C.V.},
    title = {A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild},
    year = {2020},
    isbn = {9781450379885},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3394171.3413532},
    doi = {10.1145/3394171.3413532},
    booktitle = {Proceedings of the 28th ACM International Conference on Multimedia},
    pages = {484--492},
    numpages = {9},
    keywords = {lip sync, talking face generation, video generation},
    location = {Seattle, WA, USA},
    series = {MM '20}
}
```
