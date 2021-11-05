import utils.preprocessor as up
import utils.modelling as md

if __name__ == '__main__':
    is_load = False

    dataset_dump_path = '/home/dell/PycharmProjects/attention-seq2seq/dump/dataset_params.dat'
    model_dump_path = '/home/dell/PycharmProjects/attention-seq2seq/dump/model_params.dat'
    chk_dump_path = '/home/dell/PycharmProjects/attention-seq2seq/dump/chk_params.dat'

    path_to_file = '/home/dell/PycharmProjects/attention-seq2seq/cmn-eng/cmn.txt'

    if is_load:
        dataset_params = md.params_load(dataset_dump_path)
        model_params = md.inference_model_building(dataset_params)

        chk_params = md.chk_settings(model_params)

    else:
        en, sp = up.create_dataset(path_to_file, None, True)
        print(en[-1])
        print(sp[-1])

        dataset_params, dataset = md.data_preparing(path_to_file)

        model_params = md.model_building(dataset_params, dataset)

        chk_params = md.chk_settings(model_params)

        md.params_dump(dataset_params, dataset_dump_path)

        md.trainer(dataset_params, model_params, chk_params, dataset, epochs=10)

    # 恢复检查点目录 （checkpoint_dir） 中最新的检查点
    md.restore_model(chk_params)

    md.model_visualization(model_params)
    md.translate(dataset_params, model_params, u'請把它包裝得像一個聖誕禮物。')
    md.translate(dataset_params, model_params, u'佛罗伦萨是意大利最美丽的城市。')
