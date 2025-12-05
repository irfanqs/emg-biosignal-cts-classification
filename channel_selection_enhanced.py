import numpy as np
import pandas as pd
from scipy import stats, signal
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class MultiChannelFeatureExtractor:
    """Extract komprehensif features dari multiple EMG channels (10 sadapan)"""
    
    def __init__(self, sampling_rate=1000):
        self.sampling_rate = sampling_rate
    
    def extract_time_domain_features(self, multichannel_signal):
        """
        Extract time-domain features dari SETIAP channel
        
        Args:
            multichannel_signal: (n_samples, n_channels) atau list of channels
            
        Returns:
            DataFrame with features untuk setiap channel
        """
        # Ensure format (n_samples, n_channels)
        if isinstance(multichannel_signal, list):
            channels = multichannel_signal
        elif len(multichannel_signal.shape) == 2:
            channels = [multichannel_signal[:, i] for i in range(multichannel_signal.shape[1])]
        else:
            channels = [multichannel_signal]
        
        features_list = []
        
        for ch_idx, channel in enumerate(channels):
            features = {'channel_id': ch_idx}
            
            # === TIME DOMAIN FEATURES ===
            
            # RMS (Root Mean Square) - PENTING untuk EMG
            features['rms'] = np.sqrt(np.mean(channel ** 2))
            
            # Mean Absolute Value
            features['mav'] = np.mean(np.abs(channel))
            
            # Variance & Std Dev
            features['var'] = np.var(channel)
            features['std'] = np.std(channel)
            
            # Peak to Peak
            features['peak'] = np.max(np.abs(channel))
            features['pp'] = np.max(channel) - np.min(channel)
            
            # Mean & Median
            features['mean'] = np.mean(channel)
            features['median'] = np.median(channel)
            
            # Waveform Length
            features['wl'] = np.sum(np.abs(np.diff(channel)))
            
            # Zero Crossings
            zc = np.sum(np.abs(np.diff(np.sign(channel)))) / 2
            features['zc'] = zc
            
            # Slope Sign Changes
            ssc = np.sum(np.abs(np.diff(np.sign(np.diff(channel))))) / 2
            features['ssc'] = ssc
            
            # Myopulse
            features['mp'] = np.sum(np.abs(channel) > features['std'])
            
            # Skewness & Kurtosis (Statistical moments)
            features['skewness'] = stats.skew(channel)
            features['kurtosis'] = stats.kurtosis(channel)
            
            # Signal-to-Noise Ratio (approximation)
            signal_power = np.mean(channel ** 2)
            noise_power = np.mean(np.diff(channel) ** 2)
            features['snr_approx'] = signal_power / (noise_power + 1e-10)
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def extract_frequency_domain_features(self, multichannel_signal):
        """Extract frequency-domain features (FFT based)"""
        if isinstance(multichannel_signal, list):
            channels = multichannel_signal
        elif len(multichannel_signal.shape) == 2:
            channels = [multichannel_signal[:, i] for i in range(multichannel_signal.shape[1])]
        else:
            channels = [multichannel_signal]
        
        features_list = []
        
        for ch_idx, channel in enumerate(channels):
            features = {'channel_id': ch_idx}
            
            try:
                # Compute Welch Power Spectral Density
                freqs, psd = signal.welch(
                    channel,
                    fs=self.sampling_rate,
                    nperseg=min(512, len(channel)//2),
                    window='hann'
                )
                
                psd_norm = psd / np.sum(psd)
                
                # Mean Frequency
                features['mean_freq'] = np.sum(freqs * psd_norm)
                
                # Median Frequency
                cumsum_psd = np.cumsum(psd_norm)
                median_idx = np.argmin(np.abs(cumsum_psd - 0.5))
                features['median_freq'] = freqs[median_idx]
                
                # Power dalam frequency bands
                idx_10_50 = np.where((freqs >= 10) & (freqs <= 50))[0]
                features['power_10_50Hz'] = np.sum(psd[idx_10_50])
                
                idx_50_250 = np.where((freqs >= 50) & (freqs <= 250))[0]
                features['power_50_250Hz'] = np.sum(psd[idx_50_250])
                
                idx_250 = np.where(freqs >= 250)[0]
                features['power_250Hz'] = np.sum(psd[idx_250])
                
                # Peak Frequency (dominant frequency)
                peak_idx = np.argmax(psd)
                features['peak_freq'] = freqs[peak_idx]
                
                # Total Power
                features['total_power'] = np.sum(psd)
                
                # Spectral Entropy
                entropy = -np.sum(psd_norm[psd_norm > 0] * np.log2(psd_norm[psd_norm > 0] + 1e-10))
                features['spectral_entropy'] = entropy
                
            except Exception as e:
                print(f"Warning: Frequency features error for channel {ch_idx}: {e}")
                features['mean_freq'] = 0
                features['median_freq'] = 0
                features['peak_freq'] = 0
                features['total_power'] = 0
                features['spectral_entropy'] = 0
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)


class ChannelSelector:
    """
    Statistical Channel Selection menggunakan multiple tests
    Untuk menentukan 10 sadapan mana yang paling diskriminatif untuk severity
    """
    
    def __init__(self):
        self.test_results = {}
        self.consensus_ranking = []
    
    def rank_channels_anova(self, features_list, labels, feature_name='rms'):
        """
        Rank channels menggunakan ANOVA F-test
        
        Args:
            features_list: List of feature DataFrames (satu per sample)
            labels: Label severity (0-3: non_cts, mild, moderate, severe)
            feature_name: Feature mana yang di-test (misal: 'rms')
        """
        print(f"\n{'='*70}")
        print(f"ANOVA F-Test untuk feature: {feature_name}")
        print(f"{'='*70}")
        
        n_channels = len(features_list[0]) if features_list else 0
        unique_labels = np.unique(labels)
        
        f_scores = []
        
        for ch_idx in range(n_channels):
            channel_values_by_class = []
            
            for sample_idx, label in enumerate(labels):
                feature_value = features_list[sample_idx].iloc[ch_idx][feature_name]
                channel_values_by_class.append((feature_value, label))
            
            # Group by class
            groups = [
                np.array([val for val, lbl in channel_values_by_class if lbl == cls])
                for cls in unique_labels
            ]
            
            # ANOVA test
            f_stat, p_value = stats.f_oneway(*groups)
            
            f_scores.append({
                'channel_id': ch_idx,
                'f_score': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
        
        df_anova = pd.DataFrame(f_scores).sort_values('f_score', ascending=False)
        
        print(f"\nTop 5 channels by ANOVA F-score:")
        print(df_anova[['channel_id', 'f_score', 'p_value', 'significant']].head().to_string(index=False))
        
        self.test_results[f'anova_{feature_name}'] = df_anova
        return df_anova
    
    def rank_channels_kruskal_wallis(self, features_list, labels, feature_name='rms'):
        """
        Non-parametric ranking menggunakan Kruskal-Wallis H-test
        (lebih robust untuk non-normal distributions)
        """
        print(f"\n{'='*70}")
        print(f"Kruskal-Wallis H-Test untuk feature: {feature_name}")
        print(f"{'='*70}")
        
        n_channels = len(features_list[0]) if features_list else 0
        unique_labels = np.unique(labels)
        
        h_scores = []
        
        for ch_idx in range(n_channels):
            channel_values_by_class = []
            
            for sample_idx, label in enumerate(labels):
                feature_value = features_list[sample_idx].iloc[ch_idx][feature_name]
                channel_values_by_class.append((feature_value, label))
            
            # Group by class
            groups = [
                np.array([val for val, lbl in channel_values_by_class if lbl == cls])
                for cls in unique_labels
            ]
            
            # Kruskal-Wallis test
            h_stat, p_value = stats.kruskal(*groups)
            
            h_scores.append({
                'channel_id': ch_idx,
                'h_score': h_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
        
        df_kw = pd.DataFrame(h_scores).sort_values('h_score', ascending=False)
        
        print(f"\nTop 5 channels by Kruskal-Wallis H-score:")
        print(df_kw[['channel_id', 'h_score', 'p_value', 'significant']].head().to_string(index=False))
        
        self.test_results[f'kw_{feature_name}'] = df_kw
        return df_kw
    
    def rank_channels_discrimination_power(self, features_list, labels, feature_name='rms'):
        """
        Rank by Discrimination Power (DP)
        DP = |mean_class_i - mean_class_j| / (std_class_i + std_class_j)
        """
        print(f"\n{'='*70}")
        print(f"Discrimination Power Test untuk feature: {feature_name}")
        print(f"{'='*70}")
        
        n_channels = len(features_list[0]) if features_list else 0
        unique_labels = np.unique(labels)
        
        dp_scores = []
        
        for ch_idx in range(n_channels):
            channel_values_by_class = []
            
            for sample_idx, label in enumerate(labels):
                feature_value = features_list[sample_idx].iloc[ch_idx][feature_name]
                channel_values_by_class.append((feature_value, label))
            
            # Calculate DP
            dp_total = 0
            n_comparisons = 0
            
            for i, label_i in enumerate(unique_labels):
                for j, label_j in enumerate(unique_labels):
                    if i < j:
                        vals_i = np.array([val for val, lbl in channel_values_by_class if lbl == label_i])
                        vals_j = np.array([val for val, lbl in channel_values_by_class if lbl == label_j])
                        
                        if len(vals_i) > 0 and len(vals_j) > 0:
                            mean_i, mean_j = np.mean(vals_i), np.mean(vals_j)
                            std_i, std_j = np.std(vals_i), np.std(vals_j)
                            
                            denominator = std_i + std_j + 1e-10
                            dp = np.abs(mean_i - mean_j) / denominator
                            
                            dp_total += dp
                            n_comparisons += 1
            
            dp_avg = dp_total / n_comparisons if n_comparisons > 0 else 0
            
            dp_scores.append({
                'channel_id': ch_idx,
                'dp_score': dp_avg
            })
        
        df_dp = pd.DataFrame(dp_scores).sort_values('dp_score', ascending=False)
        
        print(f"\nTop 5 channels by Discrimination Power:")
        print(df_dp[['channel_id', 'dp_score']].head().to_string(index=False))
        
        self.test_results[f'dp_{feature_name}'] = df_dp
        return df_dp
    
    def get_consensus_ranking(self, top_n_per_test=3, weights=None):
        """
        Gabungkan hasil dari semua tests untuk consensus ranking
        """
        print(f"\n{'='*70}")
        print(f"CONSENSUS CHANNEL RANKING")
        print(f"{'='*70}")
        
        if not self.test_results:
            print("No test results available!")
            return []
        
        consensus_scores = {}
        
        # Default weights jika tidak diberikan
        if weights is None:
            weights = {test_name: 1.0 for test_name in self.test_results.keys()}
        
        # Accumulate scores dari semua tests
        for test_name, results_df in self.test_results.items():
            weight = weights.get(test_name, 1.0)
            
            for rank, (_, row) in enumerate(results_df.head(top_n_per_test).iterrows()):
                ch_id = int(row['channel_id'])
                score = (top_n_per_test - rank) * weight
                
                if ch_id not in consensus_scores:
                    consensus_scores[ch_id] = 0
                consensus_scores[ch_id] += score
        
        # Sort by consensus score
        self.consensus_ranking = sorted(
            consensus_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        print(f"\nTop 10 Channels (Consensus Ranking):")
        print(f"{'Rank':<5} {'Channel ID':<12} {'Score':<10}")
        print("-" * 30)
        
        for rank, (ch_id, score) in enumerate(self.consensus_ranking[:10], 1):
            print(f"{rank:<5} {ch_id:<12} {score:.2f}")
        
        return self.consensus_ranking
    
    def plot_channel_rankings(self, top_n=10, save_path=None, figsize=(14, 8)):
        """Visualisasi channel ranking dari semua tests"""
        print(f"\nGenerating channel ranking plots...")
        
        n_tests = len(self.test_results)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for idx, (test_name, results_df) in enumerate(self.test_results.items()):
            if idx >= 4:
                break
            
            ax = axes[idx]
            
            # Get score column name
            score_col = [c for c in results_df.columns if 'score' in c][0]
            
            results_sorted = results_df.head(top_n).sort_values(score_col)
            
            # Color significant channels differently
            colors = ['green' if results_sorted.iloc[i].get('significant', False) else 'skyblue'
                     for i in range(len(results_sorted))]
            
            ax.barh(range(len(results_sorted)), results_sorted[score_col], color=colors)
            ax.set_yticks(range(len(results_sorted)))
            ax.set_yticklabels([f"Ch {int(ch_id)}" for ch_id in results_sorted['channel_id']])
            ax.set_xlabel(score_col)
            ax.set_title(f"{test_name.upper()}")
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
        
        # Hide empty subplots
        for idx in range(len(self.test_results), 4):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Channel ranking plots saved: {save_path}")
        
        plt.show()
    
    def plot_consensus_ranking(self, top_n=10, save_path=None, figsize=(12, 6)):
        """Visualisasi consensus ranking"""
        if not self.consensus_ranking:
            print("No consensus ranking available!")
            return
        
        top_channels = self.consensus_ranking[:top_n]
        ch_ids = [ch[0] for ch in top_channels]
        scores = [ch[1] for ch in top_channels]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.barh(range(len(ch_ids)), scores, color='steelblue')
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + 0.5, bar.get_y() + bar.get_height()/2,
                   f'{score:.1f}', va='center', ha='left', fontweight='bold')
        
        ax.set_yticks(range(len(ch_ids)))
        ax.set_yticklabels([f"Channel {int(ch)}" for ch in ch_ids])
        ax.set_xlabel('Consensus Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Channels - Consensus Ranking (All Statistical Tests)', 
                    fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Consensus ranking plot saved: {save_path}")
        
        plt.show()


def perform_complete_channel_analysis(multichannel_signals, labels, class_names,
                                     sampling_rate=1000, top_channels=10, 
                                     save_dir='channel_analysis_results'):
    """
    Complete pipeline untuk channel selection dengan statistical tests
    
    Args:
        multichannel_signals: Array (n_samples, n_channels, n_timepoints)
        labels: Array (n_samples,) dengan severity level
        class_names: List nama classes
        sampling_rate: EMG sampling rate
        top_channels: Berapa banyak channel yang dipilih
        save_dir: Directory untuk save results
    
    Returns:
        Dictionary dengan selected channels dan analysis results
    """
    
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("MULTI-CHANNEL EMG ANALYSIS & STATISTICAL FEATURE SELECTION")
    print("="*70)
    
    extractor = MultiChannelFeatureExtractor(sampling_rate)
    selector = ChannelSelector()
    
    # Step 1: Extract features dari semua channels
    print("\nStep 1: Extracting multi-channel features...")
    
    time_domain_features_list = []
    freq_domain_features_list = []
    
    for sample_idx, signal in enumerate(multichannel_signals):
        # Ensure signal format: (n_channels, n_timepoints)
        if len(signal.shape) == 1:
            signal = signal.reshape(1, -1)
        
        time_feat = extractor.extract_time_domain_features(signal)
        freq_feat = extractor.extract_frequency_domain_features(signal)
        
        time_domain_features_list.append(time_feat)
        freq_domain_features_list.append(freq_feat)
    
    print(f"✓ Extracted features untuk {len(multichannel_signals)} samples")
    print(f"  Time-domain features: {len(time_domain_features_list[0].columns)}")
    print(f"  Frequency-domain features: {len(freq_domain_features_list[0].columns)}")
    
    # Step 2: Statistical tests untuk ranking channels
    print("\nStep 2: Performing statistical tests...")
    
    # Test multiple features
    test_features = ['rms', 'mav', 'var', 'mean_freq', 'peak_freq']
    
    for feat in test_features:
        # ANOVA test
        selector.rank_channels_anova(time_domain_features_list, labels, feat)
        
        # Kruskal-Wallis test (non-parametric)
        selector.rank_channels_kruskal_wallis(time_domain_features_list, labels, feat)
        
        # Discrimination Power
        selector.rank_channels_discrimination_power(time_domain_features_list, labels, feat)
    
    # Step 3: Consensus ranking
    print("\nStep 3: Computing consensus ranking...")
    consensus_ranking = selector.get_consensus_ranking(top_n_per_test=3)
    
    selected_channels = [int(ch[0]) for ch in consensus_ranking[:top_channels]]
    
    print(f"\n✓ Selected top {len(selected_channels)} channels: {selected_channels}")
    
    # Step 4: Visualisasi
    print("\nStep 4: Generating visualizations...")
    
    selector.plot_channel_rankings(
        top_n=10,
        save_path=f"{save_dir}/channel_rankings_by_test.png"
    )
    
    selector.plot_consensus_ranking(
        top_n=min(10, len(selected_channels)),
        save_path=f"{save_dir}/consensus_channel_ranking.png"
    )
    
    # Step 5: Save summary report
    print("\nStep 5: Saving analysis results...")
    
    summary_df = pd.DataFrame([
        {'Rank': rank, 'Channel_ID': int(ch), 'Score': score}
        for rank, (ch, score) in enumerate(consensus_ranking[:top_channels], 1)
    ])
    
    summary_df.to_csv(f"{save_dir}/selected_channels_ranking.csv", index=False)
    
    print(f"\nResults saved to: {save_dir}")
    print("  - channel_rankings_by_test.png")
    print("  - consensus_channel_ranking.png")
    print("  - selected_channels_ranking.csv")
    
    print("\n" + "="*70)
    print("CHANNEL ANALYSIS COMPLETED!")
    print("="*70 + "\n")
    
    return {
        'selected_channels': selected_channels,
        'consensus_ranking': consensus_ranking,
        'selector': selector,
        'extractor': extractor,
        'time_domain_features': time_domain_features_list,
        'freq_domain_features': freq_domain_features_list
    }


if __name__ == "__main__":
    print("Multi-Channel EMG Feature Selection Module Loaded!")
    print("\nCapabilities:")
    print("  ✓ Extract time & frequency domain features")
    print("  ✓ ANOVA F-test")
    print("  ✓ Kruskal-Wallis H-test")
    print("  ✓ Discrimination Power analysis")
    print("  ✓ Consensus channel ranking")
    print("  ✓ Statistical visualization")