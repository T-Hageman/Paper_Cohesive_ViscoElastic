# plotting function for line plots
from matplotlib import pyplot as plt
import numpy as np
import os
import h5py

FONTSIZE = 8
LINEWIDTH = 0.5

def Ensure_Folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        
def FormatAndSave(fig, ax, SaveName, fontsize=6, top=0.97):
    plt.pause(0.1)
    
    # Total figure size: 8x6cm
    fig.set_size_inches(8/2.54, 6/2.54)
    fig.subplots_adjust(left=0.2, right=0.97, top=top, bottom=0.15)
    
    # Set the font size for all elements in the plot
    ax.title.set_fontsize(fontsize)
    ax.xaxis.label.set_fontsize(fontsize)
    ax.yaxis.label.set_fontsize(fontsize)
    ax.tick_params(axis='both', which='both', labelsize=fontsize)
    
    # Line width for all lines in the plot
    for line in ax.get_lines():
        line.set_linewidth(LINEWIDTH)
        
    # legend line width and font size
    legend = ax.get_legend()
    if legend is not None:
        for line in legend.get_lines():
            line.set_linewidth(LINEWIDTH)
        for labels in legend.get_texts():
            labels.set_fontsize(fontsize)

    plt.pause(0.1)
    fig.savefig(f"{SaveName}.png", dpi=1200)
    fig.savefig(f"{SaveName}.pdf", format='pdf', dpi=1200, metadata={"CreationDate": None})

def Plot_Lines(
    fileSets,
    line_styles,
    xticks,
    legend_labels,
    SaveName,
    xName,
    caseLabels,
    case_colors=None,
    case_markers=None,
    peak_legend_columns=1,
    peak_legend_above=False,
):
    Ensure_Folder(os.path.dirname(SaveName))
    
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    
    colour_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    
    
    for case in range(len(fileSets)):
        peakLoad = np.full(len(xticks), np.nan)
        for i, File in enumerate(fileSets[case]):
            print(f"Reading data from {File}")
            
            if os.path.exists(File):
                with h5py.File(File, 'r') as tdata:
                    t = tdata['data'][:, 0]
                    F = tdata['data'][:, 4]*1e-6 # correct for domain size
                
                if case==0:
                    ax.plot(t, F, color=colour_cycle[i], linestyle=line_styles[case], label=legend_labels[case][i])
                else: # no legend
                    ax.plot(t, F, color=colour_cycle[i], linestyle=line_styles[case], label="_nolegend_")
                    
                peakLoad[i] = np.max(F)
            else:
                print(f"File {File} does not exist. Skipping.")
        
        #ax2.plot(xticks, abs(peakLoad/peakLoad[-1]-1)*100, color=colour_cycle[case], linestyle=line_styles[case], marker='x', label=caseLabels[case])
        case_color = (
            case_colors[case]
            if case_colors is not None
            else colour_cycle[case % len(colour_cycle)]
        )
        case_marker = case_markers[case] if case_markers is not None else 'x'
        ax2.plot(
            xticks,
            peakLoad,
            color=case_color,
            linestyle=line_styles[case],
            marker=case_marker,
            markersize=3.5 if case_markers is not None else 6.0,
            markeredgewidth=0.5,
            label=caseLabels[case],
        )

    ax.set_xlabel(r'$time\;[\text{s}]$')
    ax.set_xlim([100, 5.0e4])
    ax.set_ylim([1e-3, 0.225])
    ax.set_ylabel(r'$\overline{\sigma_{yy}}\;[\text{MPa}]$')
    ax.legend(fontsize=FONTSIZE)
    FormatAndSave(fig, ax, SaveName)
    
    ax2.set_xlabel(rf'${xName}\;[\text{{m}}]$')
    ax2.set_ylabel(r'$\overline{\sigma_{yy}^{peak}}\;[\text{MPa}]$')
    #ax2.set_ylabel(r'$\Delta \overline{\sigma_{yy}^{peak}}\;[\%]$')
    #ax2.set_yscale('log')
    ax2.set_xscale('log')
    if peak_legend_above:
        ax2.legend(
            fontsize=FONTSIZE,
            ncol=peak_legend_columns,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            frameon=False,
            columnspacing=0.8,
            handletextpad=0.4,
        )
        peak_axes_top = 0.76 if len(fileSets) > 2 else 0.87
    else:
        ax2.legend(fontsize=FONTSIZE, ncol=peak_legend_columns)
        peak_axes_top = 0.97
    FormatAndSave(
        fig2,
        ax2,
        SaveName + "_PeakLoad",
        top=peak_axes_top,
    )
