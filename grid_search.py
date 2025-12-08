import itertools
from train import train

def main():
    # Grid Search Space
    learning_rates = [1e-5, 5e-5, 1e-4]
    target_kls = [0.05, 0.1, 0.2]
    
    # Total combinations: 3 * 3 = 9
    experiments = list(itertools.product(learning_rates, target_kls))
    
    print(f"Starting Grid Search with {len(experiments)} experiments...")
    
    for i, (lr, kl) in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] Running Experiment: LR={lr}, Target KL={kl}")
        
        try:
            train(
                lr=lr,
                target_kl=kl,
                steps_per_epoch=128,
                max_steps=128 * 20,
            )
            print(f"Experiment {i+1} completed successfully.")
        except Exception as e:
            print(f"Experiment {i+1} FAILED with error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
