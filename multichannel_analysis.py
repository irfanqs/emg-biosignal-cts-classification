"""
Multi-Channel EMG Feature Selection System
Extract & rank relevant channels dari 10 sadapan EMG
Determine severity dengan statistical analysis
"""

import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


class MultiChannelEMGExtractor:
    """
    Extract comprehensive features dari multiple EMG channels
    Support hingga 10 sadapan simultan
    """
    
    def __init__(self, sampling_rate=1000, n_channels=10):
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        self.feature_cache = {}
    
    def extract_time_domain_features_per_channel(self, multichannel_signal):
        """Extract time-domain features dari SETIAP channel"""
        if isinstance(multichannel_signal, list):
            channels = multichannel_signal
        elif len(multichannel_signal.shape) == 2:
            channels = [multichannel_signal[:, i] for i in range(multichannel_signal.shape[1])]
        else:
            channels = [multichannel_signal]
        
        feature_list = []
        
        for ch_idx, channel in enumerate(channels):
            features = {}
            features['channel_id'] = ch_idx
            
            # Basic statistics
            features['rms'] = np.sqrt(np.mean(channel ** 2))
            features['mav'] = np.mean(np.abs(channel))
            features['var'] = np.var(channel)
            features['std'] = np.std(channel)
            features['peak'] = np.max(np.abs(channel))
            features['mean'] = np.mean(channel)
            features['median'] = np.median(channel)
            
            # Waveform characteristics
            features['wl'] = np.sum(np.abs(np.diff(channel)))
            
            # Zero crossings
            zc = np.sum(np.abs(np.diff(np.sign(channel)))) / 2
            features['zc'] = zc
            
            # Slope sign changes
            ssc = np.sum(np.abs(np.diff(np.sign(np.diff(channel))))) / 2
            features['ssc'] = ssc
            
            # Myopulse
            features['mp'] = np.sum(np.abs(channel) > features['std'])
            
            # Skewness & Kurtosis
            features['skewness'] = stats.skew(channel)
            features['kurtosis'] = stats.kurtosis(channel)
            
            # Signal-to-noise ratio
            signal_power = np.mean(channel ** 2)
            noise_power = np.mean(np.diff(channel) ** 2)
            if noise_power > 0:
                features['snr_approx'] = signal_power / noise_power
            else:
                features['snr_approx'] = 0
            
            feature_list.append(features)
        
        return pd.DataFrame(feature_list)
    
    def extract_frequency_domain_features_per_channel(self, multichannel_signal):
        """Extract frequency-domain features menggunakan FFT dan Welch"""
        if isinstance(multichannel_signal, list):
            channels = multichannel_signal
        elif len(multichannel_signal.shape) == 2:
            channels = [multichannel_signal[:, i] for i in range(multichannel_signal.shape[1])]
        else:
            channels = [multichannel_signal]
        
        feature_list = []
        
        for ch_idx, channel in enumerate(channels):
            features = {}
            features['channel_id'] = ch_idx
            
            try:
                freqs, psd = signal.welch(
                    channel, 
                    fs=self.sampling_rate, 
                    nperseg=min(256, len(channel)//2),
                    window='hann'
                )
                
                psd_norm = psd / np.sum(psd)
                
                # Mean frequency
                features['mean_freq'] = np.sum(freqs * psd_norm)
                
                # Median frequency
                cumsum_psd = np.cumsum(psd_norm)
                median_idx = np.argmin(np.abs(cumsum_psd - 0.5))
                features['median_freq'] = freqs[median_idx]
                
                # Power bands
                idx_alpha = np.where((freqs >= 10) & (freqs <= 50))[0]
                features['power_10_50Hz'] = np.sum(psd[idx_alpha])
                
                idx_beta = np.where((freqs >= 50) & (freqs <= 250))[0]
                features['power_50_250Hz'] = np.sum(psd[idx_beta])
                
                idx_gamma = np.where(freqs >= 250)[0]
                features['power_250Hz'] = np.sum(psd[idx_gamma])
                
                # Peak frequency
                peak_idx = np.argmax(psd)
                features['peak_freq'] = freqs[peak_idx]
                features['peak_power'] = psd[peak_idx]
                
                # Spectral entropy
                entropy = -np.sum(psd_norm[psd_norm > 0] * np.log2(psd_norm[psd_norm > 0]))
                features['spectral_entropy'] = entropy
                
                # Total power
                features['total_power'] = np.sum(psd)
                
                # Power ratios
                if features['power_50_250Hz'] > 0:
                    features['power_ratio_alpha_beta'] = features['power_10_50Hz'] / features['power_50_250Hz']
                else:
                    features['power_ratio_alpha_beta'] = 0
                
            except Exception as e:
                print(f"Error computing frequency features for channel {ch_idx}: {e}")
                features['mean_freq'] = 0
                features['median_freq'] = 0
                features['spectral_entropy'] = 0
            
            feature_list.append(features)
        
        return pd.DataFrame(feature_list)
    
    def extract_complexity_features_per_channel(self, multichannel_signal):
        """Extract complexity measures: approximate entropy, etc."""
        if isinstance(multichannel_signal, list):
            channels = multichannel_signal
        elif len(multichannel_signal.shape) == 2:
            channels = [multichannel_signal[:, i] for i in range(multichannel_signal.shape[1])]
        else:
            channels = [multichannel_signal]
        
        feature_list = []
        
        for ch_idx, channel in enumerate(channels):
            features = {}
            features['channel_id'] = ch_idx
            
            # Sample Entropy (simplified)
            try:
                sample_entropy = self._calculate_sample_entropy(channel, m=2, r=0.2*np.std(channel))
                features['sample_entropy'] = sample_entropy
            except:
                features['sample_entropy'] = 0
            
            # Approximate Entropy
            try:
                approx_entropy = self._calculate_approximate_entropy(channel, m=2, r=0.2*np.std(channel))
                features['approx_entropy'] = approx_entropy
            except:
                features['approx_entropy'] = 0
            
            # Hurst Exponent
            try:
                hurst = self._calculate_hurst_exponent(channel)
                features['hurst_exponent'] = hurst
            except:
                features['hurst_exponent'] = 0.5
            
            # Signal regularity
            try:
                regularity = np.std(np.diff(channel))
                features['regularity'] = regularity
            except:
                features['regularity'] = 0
            
            feature_list.append(features)
        
        return pd.DataFrame(feature_list)
    
    def _calculate_sample_entropy(self, signal, m=2, r=None):
        """Calculate Sample Entropy"""
        if r is None:
            r = 0.2 * np.std(signal)
        
        n = len(signal)
        
        def _max_dist(x_i, x_j):
            return max(np.abs(ua - va) for ua, va in zip(x_i, x_j))
        
        templates = [signal[i:i+m] for i in range(n-m+1)]
        
        B = sum(1 for i in range(len(templates)-1) 
                for j in range(i+1, len(templates))
                if _max_dist(templates[i], templates[j]) <= r)
        
        templates_m1 = [signal[i:i+m+1] for i in range(n-m)]
        A = sum(1 for i in range(len(templates_m1)-1) 
                for j in range(i+1, len(templates_m1))
                if _max_dist(templates_m1[i], templates_m1[j]) <= r)
        
        if A == 0:
            return 0
        return -np.log(A / B) if B > 0 else 0
    
    def _calculate_approximate_entropy(self, signal, m=2, r=None):
        """Calculate Approximate Entropy"""
        if r is None:
            r = 0.2 * np.std(signal)
        
        n = len(signal)
        
        def _max_dist(x_i, x_j):
            return max(np.abs(ua - va) for ua, va in zip(x_i, x_j))
        
        patterns_m = [signal[i:i+m] for i in range(n-m+1)]
        patterns_m1 = [signal[i:i+m+1] for i in range(n-m)]
        
        C_m = sum(1 for i in range(len(patterns_m))
                  for j in range(len(patterns_m))
                  if _max_dist(patterns_m[i], patterns_m[j]) <= r)
        
        C_m1 = sum(1 for i in range(len(patterns_m1))
                   for j in range(len(patterns_m1))
                   if _max_dist(patterns_m1[i], patterns_m1[j]) <= r)
        
        if C_m1 == 0:
            return 0
        return np.log(C_m / C_m1)
    
    def _calculate_hurst_exponent(self, signal):
        """Calculate Hurst Exponent"""
        signal = np.array(signal)
        tau = []
        scales = np.unique(np.logspace(0.5, 2, 20, dtype=int))
        
        for scale in scales:
            n_chunks = len(signal) // scale
            if n_chunks == 0:
                continue
            
            chunk_rs = []
            for i in range(n_chunks):
                chunk = signal[i*scale:(i+1)*scale]
                mean_centered = chunk - np.mean(chunk)
                cumsum = np.cumsum(mean_centered)
                
                R = np.max(cumsum) - np.min(cumsum)
                S = np.std(chunk, ddof=1)
                
                if S > 0:
                    chunk_rs.append(R / S)
            
            if chunk_rs:
                tau.append(np.mean(chunk_rs))
        
        if len(tau) > 1:
            log_tau = np.log(tau)
            log_scales = np.log(scales[:len(tau)])
            hurst = np.polyfit(log_scales, log_tau, 1)[0]
            return hurst
        
        return 0.5


# ============================================================================
# 2. CHANNEL RANKING & SELECTION
# ============================================================================

class ChannelSelector:
    """Select relevant channels berdasarkan statistical significance"""
    
    def __init__(self):
        self.channel_scores = {}
        self.selected_channels = []
    
    def rank_channels_by_anova(self, all_features_per_channel, labels, top_n=5):
        """Rank channels menggunakan ANOVA F-score"""
        unique_labels = np.unique(labels)
        
        # Determine n_channels
        if isinstance(all_features_per_channel, list):
            n_channels = len(all_features_per_channel[0])
        else:
            n_channels = all_features_per_channel.shape[1]
        
        channel_f_scores = []
        
        for ch_idx in range(n_channels):
            channel_data = []
            for sample_idx, label in enumerate(labels):
                if isinstance(all_features_per_channel, list):
                    ch_features = all_features_per_channel[sample_idx].iloc[ch_idx]
                else:
                    ch_features = all_features_per_channel[sample_idx, ch_idx, :]
                
                # Use RMS as discriminator
                ch_value = ch_features['rms'] if hasattr(ch_features, '__getitem__') else np.mean(ch_features)
                channel_data.append(ch_value)
            
            channel_data = np.array(channel_data)
            
            groups = [channel_data[labels == label] for label in unique_labels]
            f_stat, p_value = stats.f_oneway(*groups)
            
            channel_f_scores.append({
                'channel_id': ch_idx,
                'f_score': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
        
        df = pd.DataFrame(channel_f_scores).sort_values('f_score', ascending=False)
        self.channel_scores['anova'] = df
        return df.head(top_n)
    
    def rank_channels_by_discrimination_power(self, all_features_per_channel, labels, top_n=5):
        """Rank channels by Discrimination Power"""
        unique_labels = np.unique(labels)
        
        if isinstance(all_features_per_channel, list):
            n_channels = len(all_features_per_channel[0])
        else:
            n_channels = all_features_per_channel.shape[1]
        
        channel_dp_scores = []
        
        for ch_idx in range(n_channels):
            channel_data = []
            for sample_idx, label in enumerate(labels):
                if isinstance(all_features_per_channel, list):
                    ch_features = all_features_per_channel[sample_idx].iloc[ch_idx]
                else:
                    ch_features = all_features_per_channel[sample_idx, ch_idx, :]
                
                ch_value = ch_features['rms'] if hasattr(ch_features, '__getitem__') else np.mean(ch_features)
                channel_data.append(ch_value)
            
            channel_data = np.array(channel_data)
            
            dp_total = 0
            n_pairs = 0
            
            for i, label_i in enumerate(unique_labels):
                for j, label_j in enumerate(unique_labels):
                    if i < j:
                        class_i = channel_data[labels == label_i]
                        class_j = channel_data[labels == label_j]
                        
                        mean_i = np.mean(class_i)
                        mean_j = np.mean(class_j)
                        std_i = np.std(class_i)
                        std_j = np.std(class_j)
                        
                        denominator = std_i + std_j
                        if denominator > 0:
                            dp = np.abs(mean_i - mean_j) / denominator
                        else:
                            dp = 0
                        
                        dp_total += dp
                        n_pairs += 1
            
            dp_avg = dp_total / n_pairs if n_pairs > 0 else 0
            
            channel_dp_scores.append({
                'channel_id': ch_idx,
                'dp_score': dp_avg,
            })
        
        df = pd.DataFrame(channel_dp_scores).sort_values('dp_score', ascending=False)
        self.channel_scores['dp'] = df
        return df.head(top_n)
    
    def rank_channels_by_kruskal_wallis(self, all_features_per_channel, labels, top_n=5):
        """Non-parametric ranking using Kruskal-Wallis test"""
        unique_labels = np.unique(labels)
        
        if isinstance(all_features_per_channel, list):
            n_channels = len(all_features_per_channel[0])
        else:
            n_channels = all_features_per_channel.shape[1]
        
        channel_kw_scores = []
        
        for ch_idx in range(n_channels):
            channel_data = []
            for sample_idx, label in enumerate(labels):
                if isinstance(all_features_per_channel, list):
                    ch_features = all_features_per_channel[sample_idx].iloc[ch_idx]
                else:
                    ch_features = all_features_per_channel[sample_idx, ch_idx, :]
                
                ch_value = ch_features['rms'] if hasattr(ch_features, '__getitem__') else np.mean(ch_features)
                channel_data.append(ch_value)
            
            channel_data = np.array(channel_data)
            
            groups = [channel_data[labels == label] for label in unique_labels]
            h_stat, p_value = stats.kruskal(*groups)
            
            channel_kw_scores.append({
                'channel_id': ch_idx,
                'h_score': h_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
        
        df = pd.DataFrame(channel_kw_scores).sort_values('h_score', ascending=False)
        self.channel_scores['kw'] = df
        return df.head(top_n)
    
    def get_consensus_channels(self, top_n_per_test=3):
        """Get consensus top channels dari multiple tests"""
        consensus_scores = {}
        
        for test_name, results in self.channel_scores.items():
            for rank, (_, row) in enumerate(results.head(top_n_per_test).iterrows()):
                ch_id = int(row['channel_id'])
                score = top_n_per_test - rank
                
                if ch_id not in consensus_scores:
                    consensus_scores[ch_id] = 0
                consensus_scores[ch_id] += score
        
        self.selected_channels = sorted(
            consensus_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return self.selected_channels
    
    def plot_channel_rankings(self, figsize=(15, 10), save_path=None):
        """Visualize channel rankings"""
        fig, axes = plt.subplots(1, len(self.channel_scores), figsize=figsize)
        
        if len(self.channel_scores) == 1:
            axes = [axes]
        
        for idx, (test_name, results) in enumerate(self.channel_scores.items()):
            ax = axes[idx]
            
            score_col = [c for c in results.columns if 'score' in c][0]
            results_sorted = results.sort_values(score_col)
            
            colors = ['green' if results_sorted.iloc[i].get('significant', False) else 'gray' 
                     for i in range(len(results_sorted))]
            
            ax.barh(range(len(results_sorted)), results_sorted[score_col], color=colors)
            ax.set_yticks(range(len(results_sorted)))
            ax.set_yticklabels([f"Ch {int(ch_id)}" for ch_id in results_sorted['channel_id']])
            ax.set_xlabel(score_col)
            ax.set_title(f"{test_name.upper()} Channel Ranking")
            ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Channel ranking plot saved: {save_path}")
        
        plt.show()


# ============================================================================
# 3. COMPLETE PIPELINE
# ============================================================================

def perform_complete_channel_analysis(multichannel_signals, labels, class_names, 
                                     sampling_rate=1000, top_channels=5, save_dir=None):
    """
    Complete multi-channel analysis pipeline
    """
    
    print("\n" + "="*80)
    print("MULTI-CHANNEL EMG ANALYSIS & CHANNEL SELECTION")
    print("="*80)
    
    extractor = MultiChannelEMGExtractor(sampling_rate)
    selector = ChannelSelector()
    
    print("\n1. Extracting multi-channel features...")
    
    time_features_list = []
    freq_features_list = []
    
    for sample_idx, signal in enumerate(multichannel_signals):
        time_feat = extractor.extract_time_domain_features_per_channel(signal)
        freq_feat = extractor.extract_frequency_domain_features_per_channel(signal)
        
        time_features_list.append(time_feat)
        freq_features_list.append(freq_feat)
    
    print(f"Extracted features for {len(multichannel_signals)} samples")
    print(f"Time features: {len(time_features_list[0].columns)}")
    print(f"Frequency features: {len(freq_features_list[0].columns)}")
    
    print("\n2. Ranking channels with statistical tests...")
    
    print("   - ANOVA test...")
    anova_ranking = selector.rank_channels_by_anova(time_features_list, labels)
    print(f"     Top 3: {list(anova_ranking['channel_id'].values[:3])}")
    
    print("   - Discrimination Power...")
    dp_ranking = selector.rank_channels_by_discrimination_power(time_features_list, labels)
    print(f"     Top 3: {list(dp_ranking['channel_id'].values[:3])}")
    
    print("   - Kruskal-Wallis test...")
    kw_ranking = selector.rank_channels_by_kruskal_wallis(freq_features_list, labels)
    print(f"     Top 3: {list(kw_ranking['channel_id'].values[:3])}")
    
    print("\n3. Consensus channel selection...")
    consensus_channels = selector.get_consensus_channels(top_n_per_test=3)
    
    selected_ch_ids = [int(ch[0]) for ch in consensus_channels[:top_channels]]
    
    print(f"   Selected top {len(selected_ch_ids)} channels: {selected_ch_ids}")
    
    for ch_id in selected_ch_ids:
        score = [s[1] for s in consensus_channels if s[0] == ch_id][0]
        print(f"      Channel {ch_id}: consensus score = {score:.2f}")
    
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        selector.plot_channel_rankings(
            save_path=f"{save_dir}/channel_rankings.png"
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        channels = [ch[0] for ch in consensus_channels]
        scores = [ch[1] for ch in consensus_channels]
        colors = ['green' if ch in selected_ch_ids else 'lightgray' for ch in channels]
        
        ax.bar(range(len(channels)), scores, color=colors)
        ax.set_xlabel('Channel ID')
        ax.set_ylabel('Consensus Score')
        ax.set_title(f'Channel Selection (Top {len(selected_ch_ids)} highlighted)')
        ax.set_xticks(range(len(channels)))
        ax.set_xticklabels([f"Ch {int(ch)}" for ch in channels], rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/channel_consensus_selection.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print("\n" + "="*80)
    print("Channel analysis completed!")
    print("="*80 + "\n")
    
    return {
        'selected_channels': selected_ch_ids,
        'consensus_scores': consensus_channels,
        'selector': selector,
        'extractor': extractor,
        'time_features': time_features_list,
        'freq_features': freq_features_list
    }


if __name__ == "__main__":
    print("Multi-Channel EMG Feature Selection Module loaded!")
    print("Berapa kode tambahan yang dibutuhkan?")
    print("- Multi-Channel Feature Extraction: ~300 baris")
    print("- Statistical Feature Selection: ~400 baris")
    print("- Channel Ranking & Selection: ~200 baris")
    print("TOTAL: ~900 baris code")