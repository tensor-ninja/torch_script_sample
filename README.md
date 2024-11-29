# Sample Embedded AI Binary

To build the program with CMake, you'll need libtorch installed. You can download it from [PyTorch](https://pytorch.org/get-started/locally/).

To build the program, run the following commands:

**Note:** before running the commands below, make sure your directory structure looks like this:

```
sample_embedded_ai_binary/
    libtorch/ <- This directory is downloaded from PyTorch
    test_ner_app/
        exported_model/ <- This file is made automatically by running `python bert_export.py`
        src/
            main.cpp
            ner_modle.cpp
            tokenizer.cpp
        include/
            ner_model.hpp
            tokenizer.hpp
        tests/
            tokenizer_test.cpp
        bert_export.py
        CMakeLists.txt
```

```bash
mkdir build
cd build
cmake -DTorch_DIR=<path_to_libtorch>/share/cmake/Torch ..
cmake --build .
```

Then, to run the program, navigate to the build directory and run:

```bash
./ner_demo
```
