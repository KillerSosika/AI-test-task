import torch

def inspect_weights(weight_path):
    print(f"\n🔍 АНАЛІЗ ВАГ: {weight_path}")
    try:
        state_dict = torch.load(weight_path, map_location='cpu')
    except Exception as e:
        print(f"❌ Помилка завантаження файлу: {e}")
        return

    has_nans = False
    has_zeros = False
    collapsed_layers = 0

    print(f"{'Назва шару (увага LoFTR)':<45} | {'Mean':>8} | {'Std (Розкид)':>12} | {'Min':>8} | {'Max':>8}")
    print("-" * 90)

    for name, tensor in state_dict.items():
        # Ігноруємо буфери батч-нормалізації
        if 'num_batches_tracked' in name:
            continue

        # Перевірка на збої пам'яті (NaN)
        if torch.isnan(tensor).any():
            has_nans = True
            
        # Перевірка на "мертві" нейрони
        if torch.all(tensor == 0):
            has_zeros = True

        # Виводимо статистику тільки для ключових шарів уваги (там де зазвичай стається колапс)
        if 'loftr_coarse.attention' in name and 'weight' in name:
            mean_val = tensor.mean().item()
            std_val = tensor.std().item()
            min_val = tensor.min().item()
            max_val = tensor.max().item()
            
            print(f"{name[:45]:<45} | {mean_val:8.4f} | {std_val:12.6f} | {min_val:8.4f} | {max_val:8.4f}")
            
            # Якщо розкид значень (Std) менше 0.001, шар скоріш за все "сколапсував" і видає константу
            if std_val < 0.001:
                collapsed_layers += 1

    if has_nans:
        print("\n❌ КРИТИЧНА ПОМИЛКА: У вагах знайдені NaN (вибух градієнтів).")
    elif has_zeros:
        print("\n❌ КРИТИЧНА ПОМИЛКА: Знайдені шари, що складаються виключно з нулів.")
    elif collapsed_layers > 0:
        print(f"\n⚠️ КОЛАПС МОДЕЛІ: {collapsed_layers} шарів уваги мають Std < 0.001. Модель видає константу (нуль точок).")
    else:
        print("\n✅ Аномалій типу NaN або нулів не знайдено. Ваги математично коректні, але логіка могла деградувати.")

if __name__ == "__main__":
    # 1. Перевіряємо зламаний фінальний файл
    inspect_weights("weights/loftr_satellite_finetuned.pth")
    
    # 2. Перевіряємо ОДИН З ПЕРШИХ файлів (судячи зі скріншота, вони в тебе є)
    # Зміни назву на ту, що реально існує в папці weights
    inspect_weights("weights/loftr_loc_1_epoch_3.pth")