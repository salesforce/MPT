# coding=utf-8
# Copyright 2020 HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Introduction to the Universal Dependencies 2.2"""

import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{nivre2018universal,
  title={Universal Dependencies 2.2},
  author={Nivre, Joakim and Abrams, Mitchell and Agi{\'c}, {\v{Z}}eljko and Ahrenberg, Lars and Antonsen, Lene and Aranzabe, Maria Jesus and Arutie, Gashaw and Asahara, Masayuki and Ateyah, Luma and Attia, Mohammed and others},
  year={2018}
}
"""

_DESCRIPTION = """\
UDPOS
"""


_DATA_OPTIONS = ['af', 'ar', 'bg', 'de', 'el', 'en', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'he', 'hi', 'hu', 'id', 'it', 'ja', 'kk', 'ko', 'mr', 'nl', 'pt', 'ru', 'ta', 'te', 'th', 'tl', 'tr', 'ur', 'vi', 'yo', 'zh']



_URL = "../../../data/udpos/"
_TRAINING_FILE = ".tsv"
_DEV_FILE = ".tsv"
_TEST_FILE = ".tsv"


class UdposConfig(datasets.BuilderConfig):
    """BuilderConfig for Udpos"""

    def __init__(self, **kwargs):
        """BuilderConfig for Udpos.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(UdposConfig, self).__init__(**kwargs)


class Udpos(datasets.GeneratorBasedBuilder):
    """Undpos dataset."""

    BUILDER_CONFIGS = [
        UdposConfig(name=config_name, version=datasets.Version("2.2.0"), description="Universal Dependencies 2.2 Dataset"
        )
        for config_name in _DATA_OPTIONS
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "pos_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "ADJ",
                                "ADP",
                                "ADV",
                                "AUX",
                                "CCONJ",
                                "DET",
                                "INTJ",
                                "NOUN",
                                "NUM",
                                "PART",
                                "PRON",
                                "PROPN",
                                "PUNCT",
                                "SCONJ",
                                "SYM",
                                "VERB",
                                "X",
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://nlp.stanford.edu/~manning/papers/nivre2020UDv2.pdf",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": f"{_URL}/train.tsv",
            "dev": f"{_URL}/valid.tsv",
            "test": f"{_URL}/test-{self.config.name}{_TEST_FILE}",
        }
        print('test config', self.config.name)
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            pos_tags = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "pos_tags": pos_tags,
                        }
                        guid += 1
                        tokens = []
                        pos_tags = []
                        chunk_tags = []
                        ner_tags = []
                else:
                    # Udpos tokens are tap separated
                    splits = line.split("\t")
                    tokens.append(splits[0])
                    pos_tags.append(splits[1])
            # last example
            yield guid, {
                "id": str(guid),
                "tokens": tokens,
                "pos_tags": pos_tags,
            }
