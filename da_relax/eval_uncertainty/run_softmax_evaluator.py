"""Run softmax evaluator."""
import os
import logging
import argparse

from predprob.pytorch.tools import flag_tools
from predprob.pytorch.tools import logging_tools

from predprob.pytorch.eval_uncertainty import softmax_evaluator
from predprob.pytorch.data import datasets
from predprob.pytorch.networks import nets

parser = argparse.ArgumentParser()
type_bool = flag_tools.type_bool

parser.add_argument('--log_root_dir', type=str,
        default='/media/yw4/hdd/log/uncertainty/eval/softmax')
parser.add_argument('--model_root_dir', type=str,
        default='/media/yw4/hdd/log/uncertainty/train_model')
parser.add_argument('--train_dataset', type=str, default='cifar5')
parser.add_argument('--test_batch', type=str, default='svhn52')
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--config_file', type=str, default='resnet_mm')
parser.add_argument('--sub_dir', type=str, default='test')
parser.add_argument('--batch_size', type=int, default=128)

FLAGS = parser.parse_args()


def main():
    train_dataset = FLAGS.train_dataset
    batch_size = FLAGS.batch_size
    test_batch = FLAGS.test_batch
    model_name = FLAGS.model
    # set up log dir
    log_dir = os.path.join(
            FLAGS.log_root_dir,
            FLAGS.train_dataset,
            FLAGS.model,
            FLAGS.config_file,
            FLAGS.sub_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = 'log_inspect_repr_{}.txt'.format(test_batch)
    logging_tools.config_logging(log_dir, filename=log_file)
    result_file = 'result_{}.csv'.format(test_batch)
    result_file = os.path.join(log_dir, result_file)
    # get model path
    model_dir = os.path.join(
            FLAGS.model_root_dir,
            FLAGS.train_dataset,
            FLAGS.model,
            FLAGS.config_file,
            FLAGS.sub_dir)
    model_path = os.path.join(model_dir, 'model.ckpt')
    if not os.path.exists(model_path):
        raise ValueError('Model checkpoint {} does not exist.'
                .format(model_path))
    # load test datasets
    logging.info('Loading test batches {}(iid) and {}(oos)...'
            .format(train_dataset, test_batch))
    test_batch_iid = datasets.get_test_batch(train_dataset)
    if test_batch == 'none':
        test_batch_oos = None
    else:
        test_batch_oos = datasets.get_test_batch(test_batch)
    n_classes = test_batch_iid.info.n_classes
    model = nets.get_model(model_name, n_classes=n_classes)
    evaluator = softmax_evaluator.SoftmaxEvaluator(
            model=model,
            model_path=model_path,
            batch_size=batch_size)
    evaluator.evaluate(
            test_batch_iid=test_batch_iid,
            test_batch_oos=test_batch_oos,
            result_file=result_file)


if __name__ == '__main__':
    main()
