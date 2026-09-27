#pragma once

#include "accuracy/accuracy_types.h"

#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace task1::accuracy {

struct FormalThreshold {
    std::string target;
    std::string dtype;
    std::string output_name;
    std::string output_kind;
    double mse_max{};
    double rmse_max{};
    double relative_l2_max{};
    double relative_linf_max{};
    double relative_floor{};
    bool numeric{};
    bool require_exact_match{};
    std::size_t max_mismatch_count{};
    std::string semantic_rule;
};

class FormalThresholdSet {
public:
    static FormalThresholdSet load(const std::filesystem::path& path);
    const FormalThreshold& require(
        const std::string& target, const std::string& dtype,
        const std::string& output_name) const;
    const std::string& id() const { return id_; }
    const std::string& sha256() const { return sha256_; }
private:
    std::string id_;
    std::string sha256_;
    std::map<std::string, FormalThreshold> entries_;
};

struct OutputAccuracyResult {
    FormalThreshold threshold;
    double mse{};
    double rmse{};
    double relative_l2{};
    double relative_linf{};
    bool exact_match{};
    std::size_t mismatch_count{};
    std::string semantic_check;
    bool pass{};
};

using AccuracyResults = std::vector<OutputAccuracyResult>;

AccuracyResults evaluate_step1_formal(
    const PipelineState&, const Step1ValidationData&, const FormalThresholdSet&);
AccuracyResults evaluate_step2_formal(
    const PipelineState&, const Step2ValidationData&, const FormalThresholdSet&);
AccuracyResults evaluate_step3_formal(
    const PipelineState&, const Step3ValidationData&, const FormalThresholdSet&);
AccuracyResults evaluate_step4_formal(
    const PipelineState&, const Step4ValidationData&, const FormalThresholdSet&);
AccuracyResults evaluate_step5_formal(
    const PipelineState&, const Step5ValidationData&, const FormalThresholdSet&);
AccuracyResults evaluate_pipeline_formal(
    const PipelineState& gpu, const PipelineState& cpu, const FormalThresholdSet&);

void merge_worst_accuracy(
    std::map<std::string, OutputAccuracyResult>& aggregate,
    const AccuracyResults& sample);
double worst_mse(const std::map<std::string, OutputAccuracyResult>&);
double worst_rmse(const std::map<std::string, OutputAccuracyResult>&);
double worst_relative_l2(const std::map<std::string, OutputAccuracyResult>&);
double worst_relative_linf(const std::map<std::string, OutputAccuracyResult>&);
bool all_accuracy_pass(const AccuracyResults&);
std::string accuracy_failure_message(const AccuracyResults&);

struct AccuracyCsvContext {
    std::string run_id;
    std::string run_type;
    std::string git_commit;
    std::string case_id;
    std::string scale_id;
    std::string backend;
    int measured_runs{};
};

void write_formal_accuracy_csv(
    const std::filesystem::path& path, const AccuracyCsvContext& context,
    const FormalThresholdSet& thresholds,
    const std::map<std::string, OutputAccuracyResult>& aggregate);

}  // namespace task1::accuracy
