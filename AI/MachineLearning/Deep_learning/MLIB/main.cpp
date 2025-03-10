#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <limits>

// ------------------------------------------------------------------
// dlib::loss_multiclass_log (Multiclass Classification Loss)
// Computes softmax probabilities and cross-entropy loss.
// Returns the average loss and a vector of predicted labels.
// ------------------------------------------------------------------
class loss_multiclass_log {
public:
    // Forward pass:
    //   - logits: a vector of samples, each sample is a vector of raw scores (one per class)
    //   - labels: ground truth class indices for each sample.
    // Returns: a pair (average_loss, predictions)
    std::pair<float, std::vector<size_t>> forward(
        const std::vector<std::vector<float>>& logits,
        const std::vector<size_t>& labels)
    {
        assert(logits.size() == labels.size());
        size_t num_samples = logits.size();
        float total_loss = 0.0f;
        std::vector<size_t> predictions(num_samples, 0);
        
        for (size_t i = 0; i < num_samples; ++i)
        {
            const auto& logit = logits[i];
            size_t num_classes = logit.size();
            // Compute softmax probabilities in a numerically stable way.
            float max_logit = *std::max_element(logit.begin(), logit.end());
            std::vector<float> exp_logits(num_classes, 0.0f);
            float sum_exp = 0.0f;
            for (size_t j = 0; j < num_classes; ++j)
            {
                exp_logits[j] = std::exp(logit[j] - max_logit);
                sum_exp += exp_logits[j];
            }
            std::vector<float> probs(num_classes, 0.0f);
            for (size_t j = 0; j < num_classes; ++j)
                probs[j] = exp_logits[j] / sum_exp;
            
            // Cross-entropy loss: -log(probability of the true class)
            float sample_loss = -std::log(probs[labels[i]] + 1e-8f);
            total_loss += sample_loss;
            // Prediction is the argmax of logits.
            size_t pred = std::distance(logit.begin(),
                                        std::max_element(logit.begin(), logit.end()));
            predictions[i] = pred;
        }
        float average_loss = total_loss / num_samples;
        return { average_loss, predictions };
    }
};

// ------------------------------------------------------------------
// dlib::loss_metric (Metric Learning Loss)
// A simplified contrastive loss: for each pair of embeddings,
// if they share the same label, we penalize the squared distance;
// if they differ, we penalize (margin - distance)_+^2.
// ------------------------------------------------------------------
class loss_metric {
public:
    float margin;
    
    // margin: the minimum desired separation between embeddings of different classes.
    loss_metric(float margin_ = 1.0f) : margin(margin_) {}

    // Forward pass:
    //   - embeddings: vector of embeddings (each is a vector<float>)
    //   - labels: corresponding class labels
    // Returns the average pairwise loss.
    float forward(const std::vector<std::vector<float>>& embeddings,
                  const std::vector<size_t>& labels)
    {
        size_t num_samples = embeddings.size();
        assert(num_samples == labels.size());
        float loss = 0.0f;
        size_t count = 0;
        // Compute pairwise losses.
        for (size_t i = 0; i < num_samples; ++i)
        {
            for (size_t j = i+1; j < num_samples; ++j)
            {
                float dist = euclidean_distance(embeddings[i], embeddings[j]);
                if (labels[i] == labels[j])
                {
                    // For similar pairs, we want embeddings to be close.
                    loss += dist * dist;
                }
                else
                {
                    // For dissimilar pairs, enforce a margin.
                    float diff = std::max(0.0f, margin - dist);
                    loss += diff * diff;
                }
                count++;
            }
        }
        return (count > 0) ? (loss / count) : 0.0f;
    }

private:
    float euclidean_distance(const std::vector<float>& a, const std::vector<float>& b)
    {
        assert(a.size() == b.size());
        float sum = 0.0f;
        for (size_t i = 0; i < a.size(); ++i)
        {
            float d = a[i] - b[i];
            sum += d * d;
        }
        return std::sqrt(sum);
    }
};

// ------------------------------------------------------------------
// dlib::loss_mmod (Object Detection Loss)
// A highly simplified version for object detection loss.
// For each predicted bounding box we try to match with a ground truth box
// using Intersection-over-Union (IoU). Unmatched predictions contribute to
// a false positive penalty and unmatched ground truths contribute to a
// false negative penalty.
// ------------------------------------------------------------------

// A simple bounding box structure.
struct box {
    float x, y, width, height;
};

// Compute Intersection over Union (IoU) for two boxes.
float intersection_over_union(const box& a, const box& b)
{
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.width, b.x + b.width);
    float y2 = std::min(a.y + a.height, b.y + b.height);
    float inter_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float union_area = a.width * a.height + b.width * b.height - inter_area;
    return (union_area > 0) ? inter_area / union_area : 0.0f;
}

class loss_mmod {
public:
    float iou_threshold;
    float false_positive_penalty;
    float false_negative_penalty;

    // iou_threshold: minimum IoU to consider a detection a match.
    // false_positive_penalty and false_negative_penalty: scalar penalties.
    loss_mmod(float iou_threshold_ = 0.5f,
              float fp_penalty = 1.0f,
              float fn_penalty = 1.0f)
        : iou_threshold(iou_threshold_),
          false_positive_penalty(fp_penalty),
          false_negative_penalty(fn_penalty)
    {}

    // Forward pass:
    //   - predicted: vector of predicted boxes (for a single image)
    //   - ground_truth: vector of ground truth boxes
    // Returns the total loss.
    float forward(const std::vector<box>& predicted,
                  const std::vector<box>& ground_truth)
    {
        float loss = 0.0f;
        std::vector<bool> gt_matched(ground_truth.size(), false);

        // For each predicted box, try to match with a ground truth box.
        for (const auto& pred : predicted)
        {
            float best_iou = 0.0f;
            size_t best_idx = 0;
            for (size_t i = 0; i < ground_truth.size(); ++i)
            {
                float iou = intersection_over_union(pred, ground_truth[i]);
                if (iou > best_iou)
                {
                    best_iou = iou;
                    best_idx = i;
                }
            }
            if (best_iou >= iou_threshold)
            {
                gt_matched[best_idx] = true;
            }
            else
            {
                // Penalize false positives.
                loss += false_positive_penalty;
            }
        }
        // Penalize false negatives (ground truths that were not detected).
        for (bool matched : gt_matched)
        {
            if (!matched)
                loss += false_negative_penalty;
        }
        return loss;
    }
};

// ------------------------------------------------------------------
// Example usage of all three loss functions.
// ------------------------------------------------------------------
int main()
{
    // Example 1: loss_multiclass_log
    {
        std::cout << "=== loss_multiclass_log ===\n";
        loss_multiclass_log loss_mc;
        // Suppose we have 3 samples and 4 classes.
        std::vector<std::vector<float>> logits = {
            {2.0f, 1.0f, 0.1f, 0.5f},
            {0.2f, 1.5f, 1.0f, 0.3f},
            {1.0f, 0.5f, 2.0f, 0.1f}
        };
        // Ground truth labels.
        std::vector<size_t> labels = {0, 1, 2};
        auto result = loss_mc.forward(logits, labels);
        std::cout << "Average loss: " << result.first << "\nPredictions: ";
        for (auto p : result.second)
            std::cout << p << " ";
        std::cout << "\n\n";
    }

    // Example 2: loss_metric
    {
        std::cout << "=== loss_metric ===\n";
        loss_metric loss_met(1.0f);
        // Suppose we have 4 embeddings of dimension 3.
        std::vector<std::vector<float>> embeddings = {
            {0.1f, 0.2f, 0.3f},
            {0.15f, 0.25f, 0.35f},
            {0.8f, 0.9f, 1.0f},
            {0.82f, 0.88f, 0.95f}
        };
        // Labels: first two belong to one class, next two to another.
        std::vector<size_t> labels = {0, 0, 1, 1};
        float met_loss = loss_met.forward(embeddings, labels);
        std::cout << "Metric loss: " << met_loss << "\n\n";
    }

    // Example 3: loss_mmod
    {
        std::cout << "=== loss_mmod ===\n";
        loss_mmod loss_mmod_layer(0.5f, 1.0f, 1.0f);
        // Predicted boxes for a single image.
        std::vector<box> predicted = {
            {10, 10, 50, 50},
            {100, 100, 40, 40}
        };
        // Ground truth boxes.
        std::vector<box> ground_truth = {
            {12, 12, 48, 48},  // should match the first predicted box.
            {200, 200, 30, 30} // missed detection.
        };
        float mmod_loss = loss_mmod_layer.forward(predicted, ground_truth);
        std::cout << "MMOD loss: " << mmod_loss << "\n\n";
    }

    return 0;
}
