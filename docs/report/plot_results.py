import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('Agg')


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BLUE = '#145a9c'
GREY = '#9b9b9b'
TEAL = '#39816c'
ORANGE = '#bf6f24'

plt.rcParams.update(
    {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Nimbus Sans', 'DejaVu Sans'],
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 9,
        'axes.linewidth': 0.6,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,
        'legend.fontsize': 7.5,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.dpi': 240,
    }
)


def rows(text):
    return [
        [re.sub(r'<[^>]+>', '', cell).replace('**', '').strip() for cell in line.strip().strip('|').split('|')]
        for line in text.splitlines()
        if line.startswith('|') and not line.startswith('|---')
    ]


def axis_style(ax):
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel('Flood-extent F1')
    ax.grid(axis='x', color='#e2e5e8', linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', width=0.6, length=3)


def save(fig, name):
    fig.savefig(HERE / f'{name}.pdf', bbox_inches='tight', pad_inches=0.035)
    fig.savefig(HERE / f'{name}.png', bbox_inches='tight', pad_inches=0.035)
    plt.close(fig)


def benchmark_plot(groups):
    methods = [
        'Combined map',
        'UN-SPIDER',
        'Otsu',
        'Radar change test',
        'Radar water model',
        'Optical water model',
        'Radar or optical water',
    ]
    columns = [1, 3, 4, 5, 6, 7, 8]
    all_values = np.array([float(groups[-1][i]) for i in columns])
    order = np.argsort(-all_values, kind='stable')
    fig, axes = plt.subplots(1, 2, figsize=(7.12, 3.25), layout='constrained')
    ax = axes[0]
    y = np.arange(len(order))
    method_colors = [BLUE, ORANGE, TEAL, '#8b699c', '#437955', '#ba6765', '#777777']
    colors = [method_colors[i] for i in order]
    ax.barh(y, all_values[order], height=0.57, color=colors)
    ax.set_yticks(y, [methods[i] for i in order])
    ax.invert_yaxis()
    for yi, value in zip(y, all_values[order], strict=True):
        ax.text(value + 0.018, yi, f'{value:.3f}', va='center', fontsize=7.5)
    axis_style(ax)
    ax.set_title('(a) All 27 events', loc='left', pad=8)

    ax = axes[1]
    labels = ['Development (12)', 'Validation (5)', 'Mountain (2)', 'Lowland (5)', 'Test (3)']
    y = np.arange(5)
    for offset, col, color, label in [
        (-0.22, 1, BLUE, 'Combined map'),
        (0, 3, ORANGE, 'UN-SPIDER'),
        (0.22, 4, TEAL, 'Otsu'),
    ]:
        ax.barh(y + offset, [float(r[col]) for r in groups[:5]], height=0.19, color=color, label=label)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    axis_style(ax)
    ax.set_title('(b) Event sets', loc='left', pad=8)
    ax.legend(
        loc='lower right',
        frameon=False,
        bbox_to_anchor=(1.0, 1.03),
        borderaxespad=0,
        handlelength=1.2,
        labelspacing=0.25,
    )
    save(fig, 'fig_benchmarks')


EVENT_NAMES = {
    'melamchi_2021g': 'Melamchi 2021',
    'kaski_2021g': 'Kaski 2021',
    'shovi_2023': 'Shovi 2023',
    'arstanbap_2024': 'Arstanbap 2024',
    'baghlan_2024': 'Baghlan 2024',
    'kathmandu_2024': 'Kathmandu 2024',
    'alex_2020': 'Storm Alex 2020',
    'jablanica_2024': 'Jablanica 2024',
    'derna_2023': 'Derna 2023',
    'slovenia_2023': 'Slovenia 2023',
    'ahr_2021': 'Ahr 2021',
    'trishuli_2026g': 'Trishuli 2026',
    'panjshir_2018': 'Panjshir 2018',
    'afg_baghlanjadid_2024': 'Baghlan-jadid 2024',
    'nepal_farwest_2020': 'Nepal Far-West 2020',
    'sunamganj_2022': 'Sunamganj 2022',
    'nepal_terai_2021': 'Nepal Terai 2021',
    'klodzko_2024': 'Kłodzko 2024',
    'cantiano_2022': 'Cantiano 2022',
    'sahuka_2025': 'Sahuka 2025',
    'larkana_2022': 'Larkana 2022',
    'kurigram_2017': 'Kurigram 2017',
    'jamalpur_2020': 'Jamalpur 2020',
    'sharaqpur_2025': 'Sharaqpur 2025',
    'thame_2024f': 'Thame GLOF 2024',
    'buner_2025f': 'Buner 2025',
    'charikar_2020f': 'Charikar 2020',
}


def event_plot(events):
    fig, axes = plt.subplots(1, 2, figsize=(7.12, 4.55), layout='constrained')
    panels = [
        [('Development', [r for r in events if r[0] == 'development'])],
        [
            (label, [r for r in events if r[0] == key])
            for key, label in [
                ('hold-out', 'Validation'),
                ('fresh mountain', 'Mountain'),
                ('fresh lowland', 'Lowland'),
                ('locked', 'Test'),
            ]
        ],
    ]
    limits = []
    for ax, panel in zip(axes, panels, strict=True):
        ypos, ticks, labels = 0, [], []
        for group, group_rows in panel:
            ordered_rows = sorted(group_rows, key=lambda r: -float(r[3]))
            ax.text(0, ypos - 0.2, group, fontsize=8, weight='bold', va='bottom')
            ypos += 0.9
            for row in ordered_rows:
                new, old = float(row[3]), float(row[5])
                ax.plot([old, new], [ypos, ypos], color='#b7bec4', linewidth=1.25)
                ax.scatter(old, ypos, s=20, facecolor='white', edgecolor=ORANGE, linewidth=0.8, zorder=3, clip_on=False)
                ax.scatter(new, ypos, s=21, color=BLUE, edgecolor='white', linewidth=0.35, zorder=4, clip_on=False)
                ax.text(max(new, old) + 0.025, ypos, f'{new:.3f}', va='center', fontsize=7)
                ticks.append(ypos)
                labels.append(EVENT_NAMES[row[1]])
                ypos += 1
            ypos += 0.65
        ax.set_yticks(ticks, labels)
        limits.append(ypos)
        axis_style(ax)
    for ax in axes:
        ax.set_ylim(max(limits) + 0.15, -0.5)
    axes[0].scatter([], [], s=21, color=BLUE, label='Combined map')
    axes[0].scatter([], [], s=20, facecolor='white', edgecolor=ORANGE, label='UN-SPIDER')
    axes[0].legend(loc='lower left', frameon=False, borderpad=0, handlelength=1, bbox_to_anchor=(-0.02, 0.015))
    save(fig, 'fig_event_scores')


def optical_plot(readme):
    data = rows(readme.split('## Flood extent', 1)[1].split('### Baselines', 1)[0])
    wanted = [
        ('12 development events, 14 October', 'Development (12)'),
        ('Same 5 events', 'Validation (5)'),
        ('5 fresh lowland', 'Lowland (5)'),
        ('2 fresh mountain', 'Mountain (2)'),
        ('3 locked', 'Test (3)'),
    ]
    selected = [(next(r for r in data if r[0].startswith(prefix)), label) for prefix, label in wanted]
    fig, ax = plt.subplots(figsize=(7.12, 2.25), layout='constrained')
    for y, (row, _label) in enumerate(selected):
        with_s2, without_s2 = float(row[1]), float(row[2])
        ax.plot([without_s2, with_s2], [y, y], color='#b7bec4', linewidth=1.5)
        ax.scatter(with_s2, y, color=BLUE, s=26, zorder=3, label='With Sentinel-2' if y == 0 else None)
        ax.scatter(
            without_s2, y, color=ORANGE, marker='s', s=23, zorder=3, label='Sentinel-2 withheld' if y == 0 else None
        )
        ax.text(without_s2 - 0.022, y, f'{without_s2:.3f}', ha='right', va='center', fontsize=7)
        ax.text(with_s2 + 0.022, y, f'{with_s2:.3f}', ha='left', va='center', fontsize=7)
    ax.set_yticks(range(len(selected)), [label for _, label in selected])
    ax.set_ylim(len(selected) - 0.5, -0.7)
    axis_style(ax)
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.02), ncol=2, frameon=False, borderaxespad=0, handlelength=1)
    save(fig, 'fig_sentinel2')


def published_model_plot():
    optical = [
        ('U-Net R34, this work', 83.3),
        ('Prithvi-EO-2.0 600M-TL', 83.1),
        ('Prithvi-EO-2.0 300M-TL', 82.6),
        ('Prithvi-100M', 80.46),
        ('U-Net, Li et al.', 84.03),
        ('SegFormer-B5', 81.89),
        ('Prithvi-CAFE', 83.41),
    ]
    radar = [
        ('U-Net R34, this work', 78.9),
        ('U-Net R50, this work', 78.8),
        ('UNet-ResNet50', 80.12),
        ('UNet-ResNet18', 79.86),
        ('UPerNet-ConvNeXt-S', 79.30),
        ('UPerNet-Swin-B', 79.28),
        ('FloodViT', 78.70),
        ('ChangeFormer', 76.96),
        ('SNUNet-CD', 74.26),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.12, 3.15), layout='constrained')
    for ax, data, xlabel, title, limits in [
        (axes[0], optical, 'Water IoU (%)', '(a) Sen1Floods11', (77.5, 87.5)),
        (axes[1], radar, 'Flood F1 (%)', '(b) Kuro Siwo', (72.5, 83.0)),
    ]:
        for y, (name, value) in enumerate(data):
            color = BLUE if 'this work' in name else ORANGE
            ax.scatter(value, y, color=color, s=24, zorder=3)
            ax.text(value + 0.25, y, f'{value:g}', va='center', fontsize=7.2)
        ax.set_yticks(range(len(data)), [name for name, _ in data])
        ax.set_ylim(len(data) - 0.5, -0.6)
        ax.set_xlim(*limits)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc='left', pad=8)
        ax.grid(axis='x', color='#e2e5e8', linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
    save(fig, 'fig_model_comparison')


def main():
    text = (ROOT / 'docs/RESULTS.md').read_text(encoding='utf-8')
    benchmark = text.split('<!-- BENCHTAB START -->', 1)[1].split('<!-- BENCHTAB END -->', 1)[0]
    group_text, event_text = benchmark.split('### Flood-extent F1 by event', 1)
    groups = rows(group_text)[1:]
    events = rows(event_text)[1:]
    assert len(groups) == 6 and len(events) == 27
    assert {r[1] for r in events} == set(EVENT_NAMES)
    np.testing.assert_allclose(
        np.mean([[float(v) for v in row[3:]] for row in events], axis=0),
        [float(v) for v in groups[-1][1:]],
        atol=0.0006,
    )
    benchmark_plot(groups)
    event_plot(events)
    optical_plot((ROOT / 'README.md').read_text(encoding='utf-8'))
    published_model_plot()


if __name__ == '__main__':
    main()
