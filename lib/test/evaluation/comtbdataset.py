import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class COMTBDataset(BaseDataset):
    # COMTB dataset
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.comtb_path
        # print(self.base_path)
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        anno_path = sequence_info['anno_path']
        ground_truth_rect = load_text(str(anno_path), delimiter=['', '\t', ','], dtype=np.float64)
        img_list = sorted([p for p in os.listdir(os.path.join(sequence_path, 'img')) if
                           os.path.splitext(p)[1] in ['.jpg', '.png', '.bmp']])
        frames = [os.path.join(sequence_path, 'img', img) for img in img_list]
        # Convert gt
        if ground_truth_rect.shape[1] > 4:
            gt_x_all = ground_truth_rect[:, [0, 2, 4, 6]]
            gt_y_all = ground_truth_rect[:, [1, 3, 5, 7]]

            x1 = np.amin(gt_x_all, 1).reshape(-1, 1)
            y1 = np.amin(gt_y_all, 1).reshape(-1, 1)
            x2 = np.amax(gt_x_all, 1).reshape(-1, 1)
            y2 = np.amax(gt_y_all, 1).reshape(-1, 1)

            ground_truth_rect = np.concatenate((x1, y1, x2 - x1, y2 - y1), 1)
        return Sequence(sequence_info['name'], frames, 'comtb', ground_truth_rect)

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_list(self):
        # NOTE: Easy_Set_Test: 214
        easy_list = ['2', '6', '7', '8', '12', '19', '20', '21', '22', '26', '28', '29', '31', '32', '33', '38',
                     '40', '41', '43', '44', '45', '47', '50', '51', '55', '57', '58', '59', '60', '61', '64', '67',
                     '69', '70', '76', '80', '81', '84', '90', '92', '96', '98', '99', '100', '101', '103', '107',
                     '108', '109', '112', '114', '115', '116', '118', '119', '120', '122', '124', '127', '128',
                     '129', '131', '132', '135', '137', '139', '140', '141', '142', '143', '145', '147', '150',
                     '151', '153', '157', '159', '160', '162', '163', '166', '167', '174', '176', '179', '188',
                     '191', '192', '195', '197', '198', '202', '203', '204', '206', '209', '210', '211', '214',
                     '220', '222', '223', '226', '237', '243', '244', '245', '247', '249', '251', '252', '254',
                     '255', '256', '258', '260', '262', '263', '264', '266', '275', '277', '278', '279', '281',
                     '282', '284', '285', '287', '288', '290', '295', '296', '306', '307', '320', '322', '323',
                     '326', '332', '335', '336', '337', '338', '340', '341', '343', '347', '348', '349', '350',
                     '352', '356', '361', '367', '370', '373', '374', '376', '381', '382', '384', '385', '386',
                     '391', '394', '396', '397', '398', '400', '403', '405', '406', '408', '409', '410', '411',
                     '412', '414', '415', '416', '417', '419', '420', '423', '424', '426', '427', '430', '434',
                     '435', '438', '439', '440', '445', '446', '447', '451', '452', '459', '462', '467', '473',
                     '479', '480', '481', '482', '483', '489', '490', '492', '495', '497', '500']

        # NOTE: Hard_Set_Test: 177
        hard_list = ['18', '182', '186', '190', '526', '538', '663', '666', '667', '669', '670', '671', '672',
                     '675', '677', '678', '679', '680', '683', '684', '692', '693', '697', '698', '701', '703',
                     '706', '709', '711', '712', '713', '714', '716', '717', '719', '720', '725', '726', '728',
                     '730', '733', '736', '739', '740', '741', '744', '747', '748', '749', '750', '754', '759',
                     '761', '762', '763', '764', '768', '769', '770', '771', '774', '776', '777', '782', '783',
                     '789', '791', '793', '795', '796', '798', '800', '801', '804', '805', '806', '807', '808',
                     '809', '810', '812', '815', '818', '820', '821', '822', '826', '827', '829', '830', '833',
                     '834', '836', '837', '839', '840', '844', '846', '847', '848', '850', '851', '852', '854',
                     '856', '859', '860', '862', '864', '865', '866', '867', '868', '872', '873', '874', '875',
                     '881', '882', '883', '884', '885', '886', '887', '890', '891', '893', '894', '897', '898',
                     '899', '902', '904', '907', '908', '909', '910', '912', '914', '916', '923', '924', '925',
                     '927', '929', '938', '941', '942', '943', '944', '945', '952', '953', '956', '957', '960',
                     '962', '963', '964', '965', '969', '971', '972', '973', '974', '975', '976', '977', '978',
                     '981', '982', '985', '986', '994', '995', '997', '1000']

        # NOTE: Joint_Set_Test
        joint_list = ['2', '6', '7', '8', '12', '19', '20', '21', '22', '26', '28', '29', '31', '32', '33', '38',
                      '40', '41', '43', '44', '45', '47', '50', '51', '55', '57', '58', '59', '60', '61', '64', '67',
                      '69', '70', '76', '80', '81', '84', '90', '92', '96', '98', '99', '100', '101', '103', '107',
                      '108', '109', '112', '114', '115', '116', '118', '119', '120', '122', '124', '127', '128',
                      '129', '131', '132', '135', '137', '139', '140', '141', '142', '143', '145', '147', '150',
                      '151', '153', '157', '159', '160', '162', '163', '166', '167', '174', '176', '179', '188',
                      '191', '192', '195', '197', '198', '202', '203', '204', '206', '209', '210', '211', '214',
                      '220', '222', '223', '226', '237', '243', '244', '245', '247', '249', '251', '252', '254',
                      '255', '256', '258', '260', '262', '263', '264', '266', '275', '277', '278', '279', '281',
                      '282', '284', '285', '287', '288', '290', '295', '296', '306', '307', '320', '322', '323',
                      '326', '332', '335', '336', '337', '338', '340', '341', '343', '347', '348', '349', '350',
                      '352', '356', '361', '367', '370', '373', '374', '376', '381', '382', '384', '385', '386',
                      '391', '394', '396', '397', '398', '400', '403', '405', '406', '408', '409', '410', '411',
                      '412', '414', '415', '416', '417', '419', '420', '423', '424', '426', '427', '430', '434',
                      '435', '438', '439', '440', '445', '446', '447', '451', '452', '459', '462', '467', '473',
                      '479', '480', '481', '482', '483', '489', '490', '492', '495', '497', '500', '18', '182',
                      '186', '190', '526', '538', '663', '666', '667', '669', '670', '671', '672', '675', '677',
                      '678', '679', '680', '683', '684', '692', '693', '697', '698', '701', '703', '706', '709',
                      '711', '712', '713', '714', '716', '717', '719', '720', '725', '726', '728', '730', '733',
                      '736', '739', '740', '741', '744', '747', '748', '749', '750', '754', '759', '761', '762',
                      '763', '764', '768', '769', '770', '771', '774', '776', '777', '782', '783', '789', '791',
                      '793', '795', '796', '798', '800', '801', '804', '805', '806', '807', '808', '809', '810',
                      '812', '815', '818', '820', '821', '822', '826', '827', '829', '830', '833', '834', '836',
                      '837', '839', '840', '844', '846', '847', '848', '850', '851', '852', '854', '856', '859',
                      '860', '862', '864', '865', '866', '867', '868', '872', '873', '874', '875', '881', '882',
                      '883', '884', '885', '886', '887', '890', '891', '893', '894', '897', '898', '899', '902',
                      '904', '907', '908', '909', '910', '912', '914', '916', '923', '924', '925', '927', '929',
                      '938', '941', '942', '943', '944', '945', '952', '953', '956', '957', '960', '962', '963',
                      '964', '965', '969', '971', '972', '973', '974', '975', '976', '977', '978', '981', '982',
                      '985', '986', '994', '995', '997', '1000']

        sequence_list = joint_list
        # NOTE: for visualization
        # sequence_list = ['69']
        sequence_info_list = []
        for i in range(len(sequence_list)):
            sequence_info = {}
            sequence_info["name"] = sequence_list[i]
            dataset_path = '/home/pod/shared-nvme/dataset/'
            sequence_info["path"] = dataset_path + sequence_info["name"]
            # sequence_info["startFrame"] = int('1')
            # print(end_frame[i])
            # sequence_info["endFrame"] = end_frame[i]

            # sequence_info["nz"] = int('6')
            # sequence_info["ext"] = 'jpg'
            sequence_info["anno_path"] = sequence_info["path"] + '/groundtruth_rect.txt'
            # sequence_info["object_class"] = 'person'
            sequence_info_list.append(sequence_info)
        return sequence_info_list
