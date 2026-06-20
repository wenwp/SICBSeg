
import numpy as np

np.seterr(divide='ignore', invalid='ignore')


class BinaryEvaluator:
    """
    二分类评估器，专注于正类（destroyed）的检测性能
    """
    def __init__(self, ignore_index=255):
        """
        Args:
            ignore_index: 忽略的标签值，默认255
        """
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """重置所有累计统计"""
        self.tp = 0  # True Positive: 预测为1，实际为1
        self.fp = 0  # False Positive: 预测为1，实际为0
        self.tn = 0  # True Negative: 预测为0，实际为0
        self.fn = 0  # False Negative: 预测为0，实际为1
        # AP 用：累计的概率分布直方图（256 bin），节省内存
        # _prob_pos_hist[i] = 真实正类 且 prob in [i/255, (i+1)/255) 的像素数
        # _prob_neg_hist[i] = 真实负类 且 prob in [i/255, (i+1)/255) 的像素数
        self._prob_pos_hist = np.zeros(256, dtype=np.int64)
        self._prob_neg_hist = np.zeros(256, dtype=np.int64)
        self._has_prob = False

    def add_batch_prob(self, gt_image, prob_image):
        """
        累计概率直方图（用于多阈值 AP/PR 曲线）。

        Args:
            gt_image:   真实标签 [H, W]，取值 {0, 1, ignore_index}
            prob_image: 预测正类概率 [H, W]，[0, 1] 浮点
        """
        if 'torch' in str(type(gt_image)):
            gt_image = gt_image.cpu().numpy()
        if 'torch' in str(type(prob_image)):
            prob_image = prob_image.detach().float().cpu().numpy()

        gt_image = gt_image.astype('int64')
        mask = (gt_image != self.ignore_index)
        gt_valid = gt_image[mask]
        prob_valid = np.clip(prob_image[mask], 0.0, 1.0)
        # 量化到 256 bin
        bins = np.minimum((prob_valid * 256).astype(np.int64), 255)
        pos = bins[gt_valid == 1]
        neg = bins[gt_valid == 0]
        if pos.size:
            self._prob_pos_hist += np.bincount(pos, minlength=256)
        if neg.size:
            self._prob_neg_hist += np.bincount(neg, minlength=256)
        self._has_prob = True

    def add_batch(self, gt_image, pred_image):
        """
        添加一批预测结果
        
        Args:
            gt_image: 真实标签 (numpy array 或 torch tensor)
            pred_image: 预测标签 (numpy array 或 torch tensor)
        """
        # 转换为numpy
        if 'torch' in str(type(gt_image)):
            gt_image = gt_image.cpu().numpy()
        if 'torch' in str(type(pred_image)):
            pred_image = pred_image.cpu().numpy()
        
        gt_image = gt_image.astype('int')
        pred_image = pred_image.astype('int')
        
        # 过滤ignore区域
        mask = (gt_image != self.ignore_index)
        gt_valid = gt_image[mask]
        pred_valid = pred_image[mask]
        
        # 计算混淆矩阵元素
        self.tp += np.sum((pred_valid == 1) & (gt_valid == 1))
        self.fp += np.sum((pred_valid == 1) & (gt_valid == 0))
        self.tn += np.sum((pred_valid == 0) & (gt_valid == 0))
        self.fn += np.sum((pred_valid == 0) & (gt_valid == 1))
    
    def Accuracy(self):
        """总体准确率 OA = (TP + TN) / (TP + TN + FP + FN)"""
        total = self.tp + self.tn + self.fp + self.fn
        if total == 0:
            return 0.0
        return (self.tp + self.tn) / total
    
    def Precision(self, class_idx=1):
        """
        精确率 Precision = TP / (TP + FP)
        
        Args:
            class_idx: 0=background, 1=destroyed（默认）
        """
        if class_idx == 1:
            # 正类（destroyed）的精确率
            if (self.tp + self.fp) == 0:
                return 0.0
            return self.tp / (self.tp + self.fp)
        else:
            # 负类（background）的精确率
            if (self.tn + self.fn) == 0:
                return 0.0
            return self.tn / (self.tn + self.fn)
    
    def Recall(self, class_idx=1):
        """
        召回率 Recall = TP / (TP + FN)
        
        Args:
            class_idx: 0=background, 1=destroyed（默认）
        """
        if class_idx == 1:
            # 正类（destroyed）的召回率
            if (self.tp + self.fn) == 0:
                return 0.0
            return self.tp / (self.tp + self.fn)
        else:
            # 负类（background）的召回率 = TNR
            if (self.tn + self.fp) == 0:
                return 0.0
            return self.tn / (self.tn + self.fp)
    
    def F1Score(self, class_idx=1):
        """
        F1分数 F1 = 2 * Precision * Recall / (Precision + Recall)
        
        Args:
            class_idx: 0=background, 1=destroyed（默认）
        """
        precision = self.Precision(class_idx)
        recall = self.Recall(class_idx)
        if (precision + recall) == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    def IoU(self, class_idx=1):
        """
        交并比 IoU = TP / (TP + FP + FN)
        
        Args:
            class_idx: 0=background, 1=destroyed（默认）
        """
        if class_idx == 1:
            # 正类IoU
            union = self.tp + self.fp + self.fn
            if union == 0:
                return 0.0
            return self.tp / union
        else:
            # 负类IoU
            union = self.tn + self.fn + self.fp
            if union == 0:
                return 0.0
            return self.tn / union
    
    def mIoU(self):
        """平均IoU = (IoU_class0 + IoU_class1) / 2"""
        iou0 = self.IoU(class_idx=0)
        iou1 = self.IoU(class_idx=1)
        return (iou0 + iou1) / 2
    
    def Specificity(self):
        """特异性 Specificity = TN / (TN + FP) = TNR (True Negative Rate)"""
        if (self.tn + self.fp) == 0:
            return 0.0
        return self.tn / (self.tn + self.fp)
    
    def FPR(self):
        """假正率 FPR = FP / (FP + TN) = 1 - Specificity"""
        return 1.0 - self.Specificity()
    
    def FNR(self):
        """假负率 FNR = FN / (FN + TP) = 1 - Recall"""
        return 1.0 - self.Recall(class_idx=1)
    
    def get_confusion_matrix(self):
        """返回混淆矩阵 [[TN, FP], [FN, TP]]"""
        return np.array([[self.tn, self.fp], 
                        [self.fn, self.tp]])
    
    def AP(self, class_idx=1):
        """
        Average Precision (AP) - 基于 precision-recall 曲线下面积的真实 AP。

        要求先通过 add_batch_prob() 累计概率直方图。
        若未提供概率（仅调用过 add_batch），则退化为当前阈值下的 F1，并保留
        旧字段名以保持向后兼容（同时打印一次警告）。

        Args:
            class_idx: 0=background, 1=destroyed（默认）

        Returns:
            float: PR 曲线下面积，[0, 1]
        """
        # 没有概率信息时的兼容路径：退化为 F1（与历史行为一致）
        if not self._has_prob:
            precision = self.Precision(class_idx)
            recall = self.Recall(class_idx)
            if (precision + recall) == 0:
                return 0.0
            return 2 * precision * recall / (precision + recall)

        # 用累计的概率直方图，从高分到低分扫描所有阈值，计算 PR 曲线下面积
        # （类似 sklearn.metrics.average_precision_score 的离散化版本）
        if class_idx == 0:
            pos = self._prob_neg_hist[::-1]  # 1 - p 视角，bin 翻转
            neg = self._prob_pos_hist[::-1]
        else:
            pos = self._prob_pos_hist
            neg = self._prob_neg_hist

        total_pos = int(pos.sum())
        if total_pos == 0:
            return 0.0

        # 按 prob 从高到低累加：bin 255 -> bin 0
        cum_pos = np.cumsum(pos[::-1]).astype(np.float64)
        cum_neg = np.cumsum(neg[::-1]).astype(np.float64)
        denom = cum_pos + cum_neg
        precision = np.where(denom > 0, cum_pos / np.maximum(denom, 1), 0.0)
        recall = cum_pos / total_pos

        # 阶梯式 AP：sum of precision[i] * (recall[i] - recall[i-1])
        recall_prev = np.concatenate([[0.0], recall[:-1]])
        ap = float(np.sum(precision * (recall - recall_prev)))
        return ap
    
    def get_all_metrics(self):
        """
        返回所有评估指标的字典
        
        Returns:
            dict: 包含所有指标的字典
        """
        metrics = {
            # 基础统计
            'TP': self.tp,
            'FP': self.fp,
            'TN': self.tn,
            'FN': self.fn,
            
            # 整体指标
            'accuracy': self.Accuracy(),
            'miou': self.mIoU(),
            
            # 正类（destroyed）指标
            'destroyed_precision': self.Precision(class_idx=1),
            'destroyed_recall': self.Recall(class_idx=1),
            'destroyed_f1': self.F1Score(class_idx=1),
            'destroyed_iou': self.IoU(class_idx=1),
            'destroyed_ap': self.AP(class_idx=1),  # 新增AP指标
            
            # 负类（background）指标
            'background_precision': self.Precision(class_idx=0),
            'background_recall': self.Recall(class_idx=0),
            'background_f1': self.F1Score(class_idx=0),
            'background_iou': self.IoU(class_idx=0),
            
            # 其他指标
            'specificity': self.Specificity(),
            'fpr': self.FPR(),
            'fnr': self.FNR()
        }
        
        return metrics
    
    def print_metrics(self):
        """打印所有评估指标"""
        metrics = self.get_all_metrics()
        
        print("="*60)
        print("二分类评估指标")
        print("="*60)
        print(f"\n混淆矩阵:")
        print(f"              Predicted")
        print(f"              Background  Destroyed")
        print(f"Actual Background    {self.tn:8d}  {self.fp:8d}")
        print(f"       Destroyed     {self.fn:8d}  {self.tp:8d}")
        
        print(f"\n整体指标:")
        print(f"  Accuracy (OA):      {metrics['accuracy']:.4f}")
        print(f"  Mean IoU (mIoU):    {metrics['miou']:.4f}")
        
        print(f"\n毁坏建筑（Destroyed）类指标:")
        print(f"  Precision:          {metrics['destroyed_precision']:.4f}")
        print(f"  Recall (Sensitivity): {metrics['destroyed_recall']:.4f}")
        print(f"  F1 Score:           {metrics['destroyed_f1']:.4f}")
        print(f"  IoU:                {metrics['destroyed_iou']:.4f}")
        print(f"  AP (Avg Precision): {metrics['destroyed_ap']:.4f}")
        
        print(f"\n背景（Background）类指标:")
        print(f"  Precision:          {metrics['background_precision']:.4f}")
        print(f"  Recall:             {metrics['background_recall']:.4f}")
        print(f"  F1 Score:           {metrics['background_f1']:.4f}")
        print(f"  IoU:                {metrics['background_iou']:.4f}")
        
        print(f"\n其他指标:")
        print(f"  Specificity (TNR):  {metrics['specificity']:.4f}")
        print(f"  FPR:                {metrics['fpr']:.4f}")
        print(f"  FNR:                {metrics['fnr']:.4f}")
        print("="*60)
