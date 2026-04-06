from capability_cartography_layer4.case_study import run_minimal_case_study


def main() -> None:
    result = run_minimal_case_study()
    registration = result["forecast_registration"]
    masking = result["checkpoint_metrics"]["masking_analysis"]
    causal = result["causal_validation"]

    print("=" * 80)
    print("CCL4 MINIMAL END-TO-END CASE STUDY")
    print("=" * 80)
    print(f"Forecast: {registration['forecast_type']} [{registration['regime_label']}]")
    print(f"Observed curve: {result['checkpoint_metrics']['observed_curve_class']}")
    print(f"Masking supported: {masking['masking_supported']}")
    print(f"Fine RMSE drop before jump: {masking['fine_rmse_drop_before_jump']}")
    print(f"Fourier signature gain before jump: {masking['fourier_signature_gain_before_jump']}")
    print(f"Ablation accuracy drop: {causal['accuracy_drop_from_ablation']}")
    print(f"Restoration recovery: {causal['accuracy_recovery_after_restoration']}")
    print(f"Artifacts: {result['artifact_dir']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
