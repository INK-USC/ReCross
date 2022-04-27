import os
import json

from torch.utils import data
from .base_datamanager import MyQADataset, MyDataLoader
from .eval_metrics import METRICS, evaluate_func
from IPython import embed


class GeneralDataset(object):

    def __init__(self, logger, args, data_path, is_training, task_name, given_data=None):
        # should give the tasks used in this split in the var "tasks"
        self.data_path = data_path

        self.data = []
        self.task_name = task_name
        if given_data:
            self.data = given_data
        else:
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data path {data_path} does not exist.")
            try:
                with open(data_path) as fin:
                    lines = fin.readlines()

                # train_examples = []
                for line in lines:
                    # d = line.strip().split("\t")
                    # self.data.append((d[0], d[1:]))
                    d = json.loads(line)
                    self.data.append((d["input"], d["output"], d["id"]))
            except:
                with open(data_path) as fin:
                    lines = fin.read()
                    d = json.loads(lines)
                    for line in d:
                        self.data.append((line[0], line[1], line[2]))

        self.is_training = is_training
        self.load = not args.debug if hasattr(args, "debug") else True
        self.logger = logger
        self.args = args
        self.max_input_length = self.args.max_input_length

        try:
            self.metric = METRICS[self.task_name]
        except KeyError:
            self.metric = "EM|SoftEM"
            self.logger.warn(
                f"Metric for {self.task_name} not found, setting the metric to EM and SoftEM")
        self.tokenizer = None
        self.dataset = None
        self.dataloader = None
        self.cache = None

        self.gen_early_stop = True # TODO: double check

        if self.task_name == "story_cloze-2016":
            self.gen_early_stop = False # TODO: double check

        self.logger.debug(f"task_name={self.task_name}")
        self.logger.debug(f"gen_early_stop={self.gen_early_stop}")

    def __len__(self):
        return len(self.data)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def decode_batch(self, tokens):
        return [self.decode(_tokens) for _tokens in tokens]

    def flatten(self, answers):
        new_answers, metadata = [], []
        for answer in answers:
            metadata.append((len(new_answers), len(new_answers)+len(answer)))
            new_answers += answer
        return new_answers, metadata

    def load_dataset(self, tokenizer, do_return=False, skip_cache=False, quiet=False):
        self.tokenizer = tokenizer
        postfix = "prepro" + tokenizer.__class__.__name__.replace("zer", "zed")

        if not skip_cache:
            preprocessed_path = os.path.join(
                "/".join(self.data_path.split("/")[:-1]),
                self.data_path.split("/")[-1].replace(".jsonl", "-{}.json".format(postfix)))
            self.logger.info(f"preprocessed_path={preprocessed_path}")
        if not skip_cache and self.load and os.path.exists(preprocessed_path):
            # load preprocessed input
            self.logger.info(
                "Loading pre-tokenized data from {}".format(preprocessed_path))
            with open(preprocessed_path, "r") as f:
                input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, \
                    metadata = json.load(f)

        else:
            if not quiet:
                self.logger.info(
                    "Start tokenizing ... {} instances".format(len(self.data)))

            inputs = []
            outputs = []
            uuids = []

            for dp in self.data:
                # if len(dp) != 3:
                #     print(self.data)
                #     print(len(dp))
                #     print(dp)
                # Add the task name to the input
                # inputs.append(" [{}] {}".format(self.task_name, dp[0]))
                inputs.append(dp[0])
                outputs.append(dp[1])  # is a list
                uuids.append(dp[2])

            if not quiet:
                self.logger.info("Printing 3 examples")
                for i in range(3):
                    self.logger.info(inputs[i])
                    self.logger.info(outputs[i])

            outputs, metadata = self.flatten(outputs)  # what is metadata?

            if self.args.do_lowercase:
                inputs = [input0.lower() for input0 in inputs]
                outputs = [output0.lower() for output0 in outputs]
            if self.args.append_another_bos:
                inputs = ["<s> "+input0 for input0 in inputs]
                outputs = ["<s> " + output0 for output0 in outputs]

            if not quiet:
                self.logger.info("Tokenizing Input ...")
            tokenized_input = tokenizer.batch_encode_plus(inputs,
                                                          pad_to_max_length=True,
                                                          max_length=self.max_input_length,
                                                          truncation=True)
            if not quiet:
                self.logger.info("Tokenizing Input ... Done!")
                self.logger.info("Tokenizing Output ...")

            try:
                tokenized_output = tokenizer.batch_encode_plus(outputs,
                                                               pad_to_max_length=True,
                                                               max_length=self.args.max_output_length,
                                                               truncation=True)
            except Exception as e:
                for idx, output in enumerate(outputs):
                    if output == '':

                        self.logger.info('===================================')
                        self.logger.info(idx)
                        self.logger.info('----------------')
                        self.logger.info(inputs[idx])
                        self.logger.info('----------------')
                        self.logger.info(outputs[idx])
                        self.logger.info('----------------')
                        self.logger.info(uuids[idx])

                exit(-1)

            if not quiet:
                self.logger.info("Tokenizing Output ... Done!")
            input_ids, attention_mask = tokenized_input["input_ids"], tokenized_input["attention_mask"]
            decoder_input_ids, decoder_attention_mask = tokenized_output[
                "input_ids"], tokenized_output["attention_mask"]

            if self.load and not skip_cache:
                preprocessed_data = [input_ids, attention_mask,
                                     decoder_input_ids, decoder_attention_mask,
                                     metadata]
                self.logger.info("Save preprocessed data ...")
                with open(preprocessed_path, "w") as f:
                    json.dump([input_ids, attention_mask,
                               decoder_input_ids, decoder_attention_mask,
                               metadata], f)
                self.logger.info("Save preprocessed data ... Done!")

        assert len(uuids) == len(input_ids)  # make sure

        self.dataset = MyQADataset(input_ids, attention_mask,
                                   decoder_input_ids, decoder_attention_mask,
                                   in_metadata=None, out_metadata=metadata,
                                   is_training=self.is_training, uuids=uuids)
        if not quiet:
            self.logger.info("Loaded {} examples from data".format(
                len(self.dataset)))

        if do_return:
            return self.dataset

    def load_dataloader(self, do_return=False, is_training="self"):
        if is_training == "self":
            is_training = self.is_training
        self.dataloader = MyDataLoader(
            self.args, self.dataset, is_training)
        if do_return:
            return self.dataloader

    def evaluate(self, predictions):
        assert len(predictions) == len(self), (len(predictions), len(self))
        predictions = [prediction.strip() for prediction in predictions]
        return evaluate_func(predictions, self.data, self.metric)

    def save_predictions(self, predictions, path_to_save=None, prefix=None):
        assert len(predictions) == len(self), (len(predictions), len(self))

        predictions = ['n/a' if len(prediction.strip()) ==
                       0 else prediction for prediction in predictions]
        prediction_text = [{'id': truth[2], 'input': truth[0], 'truth': truth[1],
                            'prediction': pd} for pd, truth in zip(predictions, self.data)]
        if path_to_save:
            save_path = path_to_save
        elif prefix:
            save_path = os.path.join(
                self.args.output_dir, f"{self.args.run_name}_{prefix}_predictions.json")
        else:
            save_path = os.path.join(
                self.args.output_dir, "{}_predictions.json".format(self.args.run_name))
        os.makedirs(self.args.output_dir, exist_ok=True)

        # Save predictions to a JSON file
        with open(save_path, 'w') as fp:
            fp.write(json.dumps(prediction_text, indent=4))

        self.logger.info("Saved prediction in {}".format(save_path))
