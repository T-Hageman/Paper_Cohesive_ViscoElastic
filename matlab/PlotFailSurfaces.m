function metadata = PlotFailSurfaces()
% Independent publication figures for the calibrated r2_2L criterion in IceCubes.
% Run with: addpath('matlab'); PlotFailSurfaces
% Sources: shared/fracture_relations.h and compute_strength_surface_point.
% Plot the degree-one strength envelope and verify it independently against
% the regularized finite-probe construction used by the C++ plotting helper.
% Panel (b) also includes inline-labelled, constant creep-rate contours.
% Each canvas: 90 x 95 mm; normalized stresses; vector EPS/PDF.
% Panel letters and individual captions are supplied by LaTeX subfigures.
% Elsevier artwork sizing/types guidance checked 21 October 2026:
% https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-sizing

%% Material and output settings
material = struct('FailureType', "r2_2L", ...
    'Youngs', 9.5e9, 'Poisson', 0.35, 'f_t', 0.4e6, 'f_c', 0.6e6);
assert(material.FailureType == "r2_2L", 'This publication figure uses the r2_2L criterion.');
assert(material.f_c > material.f_t && material.f_t > 0, ...
    'The calibrated r2_2L criterion requires f_c > f_t > 0.');
material.Shear = material.Youngs / (2 * (1 + material.Poisson));
material.Bulk = material.Youngs / (3 * (1 - 2 * material.Poisson));
material.Lame = material.Bulk - 2 * material.Shear / 3;
material.stressToStrain = inv(material.Lame * ones(3) + 2 * material.Shear * eye(3));
material.stressToStrain2D = inv([material.Lame + 2 * material.Shear, material.Lame; ...
    material.Lame, material.Lame + 2 * material.Shear]);
material.bulk2D = material.Lame + material.Shear;
figureDirectory = fileparts(mfilename('fullpath'));
outputStems = struct('threeD', fullfile(figureDirectory, 'failure_surface_3d'), ...
    'twoD', fullfile(figureDirectory, 'failure_surface_2d'));
metadata = validateCriterion(material);
metadata.material = material;
metadata.width_mm = 90;
metadata.height_mm = 95;
metadata.output_stems = outputStems;
metadata.stress_normalization = 'f_t';

%% Smooth parametric 3-D surface: compression cylinder and tensile cap
ratio = material.f_c / material.f_t;
volumetricRatio = 1 / sqrt(1 - ratio^(-2));
theta = linspace(0, 2 * pi, 121);
meanStress = unique([linspace(-1.35, 0, 31), ...
    volumetricRatio / 3 * sin(linspace(0, pi / 2, 35))], 'stable');
[thetaGrid, meanGrid] = meshgrid(theta, meanStress);
qOverFt = ratio * sqrt(max(0, 1 - (3 * max(meanGrid, 0) / volumetricRatio).^2));
radius = sqrt(2 / 3) * qOverFt;
basis1 = [1; -1; 0] / sqrt(2);
basis2 = [1; 1; -2] / sqrt(6);
principal = zeros([size(meanGrid), 3]);
for component = 1:3
    principal(:, :, component) = meanGrid + radius .* ...
        (basis1(component) * cos(thetaGrid) + basis2(component) * sin(thetaGrid));
end
surfaceError = max(abs(r2Indicator(reshape(principal, [], 3) * material.f_t, material, false)));
assert(surfaceError < 1e-12, 'Parametric 3-D surface is inconsistent with the criterion.');
metadata.parametric_surface_error = surfaceError;

%% Intrinsic 2-D envelope and the 3-D plane-stress section
axisLimits = [-2.5, 1.5];
axisValues = linspace(axisLimits(1), axisLimits(2), 801);
[s1, s2] = meshgrid(axisValues);
sectionStress = [s1(:), s2(:), zeros(numel(s1), 1)] * material.f_t;
true2D = reshape(r2Indicator(sectionStress, material, true), size(s1));
section3D = reshape(r2Indicator(sectionStress, material, false), size(s1));
creepLevels = [0.5, 1.0, 1.5];
metadata.creep_levels = creepLevels;
metadata.creep_normalization = 'chi = (epsdot_eq / A)^(1/n) / f_t';

%% Final-size vector artwork
blue = [0, 114, 178] / 255;
% Red rather than orange: orange marks the opening-only criterion in Fig. 2.
red = [200, 20, 40] / 255;
fig3 = publicationFigure(metadata.width_mm / 10, metadata.height_mm / 10);
cleanupFigure3 = onCleanup(@() close(fig3));
ax3 = axes(fig3, 'Units', 'centimeters', 'Position', [0.85, 2.15, 7.65, 6.8]);
hold(ax3, 'on');
% View from the tensile octant, as in PlotCliffFailSurfaces.m. The surface is
% convex, so the vector renderer's depth sorting is replaced by drawing the
% facets from back to front and omitting the hidden parts of the grid lines.
viewAngles = [115, 20];
camera = [sind(viewAngles(1)) * cosd(viewAngles(2)); ...
    -cosd(viewAngles(1)) * cosd(viewAngles(2)); sind(viewAngles(2))];
set(ax3, 'SortMethod', 'childorder');
% Outward normal of the surface of revolution: radial unit vector tilted
% along the hydrostatic axis by the slope of the meridian.
hydrostatic = ones(3, 1) / sqrt(3);
radial = basis1 * cos(thetaGrid(:)).' + basis2 * sin(thetaGrid(:)).';
[~, slope] = gradient(radius, theta, meanStress);
normals = radial - hydrostatic * (slope(:).' / sqrt(3));
normals(:, radius(:) < 1e-12) = repmat(hydrostatic, 1, nnz(radius(:) < 1e-12));
visible = reshape(camera.' * normals >= 0, size(radius));
% Lines continue one sample past the outline, so that they end on it.
visible = visible | circshift(visible, 1, 1) | circshift(visible, -1, 1) ...
    | circshift(visible, 1, 2) | circshift(visible, -1, 2);
faceColor = [0.78, 0.88, 0.94];
[faces, vertices] = surf2patch(principal(:, :, 1), principal(:, :, 2), principal(:, :, 3));
[~, order] = sort(mean(reshape(vertices(faces, :) * camera, size(faces)), 2));
patch(ax3, 'Faces', faces(order, :), 'Vertices', vertices, ...
    'FaceColor', faceColor, 'EdgeColor', faceColor, 'LineWidth', 0.3);
% Sparse curves reveal curvature without transparency or rasterized lighting.
hiddenNaN = principal;
for component = 1:3
    layer = hiddenNaN(:, :, component); layer(~visible) = NaN;
    hiddenNaN(:, :, component) = layer;
end
for column = 1:10:size(principal, 2)
    plot3(ax3, hiddenNaN(:, column, 1), hiddenNaN(:, column, 2), ...
        hiddenNaN(:, column, 3), 'Color', blue, 'LineWidth', 0.45);
end
for row = [1, 11, 21, 31, 42, 52, 61]
    if row <= size(principal, 1)
        plot3(ax3, hiddenNaN(row, :, 1), hiddenNaN(row, :, 2), ...
            hiddenNaN(row, :, 3), 'Color', blue, 'LineWidth', 0.55);
    end
end
% The hydrostatic axis leaves the surface at the tip of the cap.
tip = max(meanStress);
hydrostaticHandle = plot3(ax3, [tip, axisLimits(2)], [tip, axisLimits(2)], ...
    [tip, axisLimits(2)], 'k--', 'LineWidth', 0.9);
axis(ax3, 'equal');
xlim(ax3, axisLimits); ylim(ax3, axisLimits); zlim(ax3, axisLimits);
view(ax3, viewAngles);
% Keep [-] outside math mode so its single hyphen is not a math minus.
xlabel(ax3, '$\sigma_1 / f_{\mathrm{t}}$ [-]', 'Interpreter', 'latex', 'FontSize', 10);
ylabel(ax3, '$\sigma_2 / f_{\mathrm{t}}$ [-]', 'Interpreter', 'latex', 'FontSize', 10);
zlabel(ax3, '$\sigma_3 / f_{\mathrm{t}}$ [-]', 'Interpreter', 'latex', 'FontSize', 10);
styleAxes(ax3);
% Fix the labels beside the projected axes; automatic 3-D LaTeX label
% placement leaves excessive space below the tick labels in vector exports.
set(ax3.XLabel, 'Units', 'normalized', 'Position', [0.93, -0.04, 0], ...
    'Rotation', 37, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
set(ax3.YLabel, 'Units', 'normalized', 'Position', [0.31, -0.10, 0], ...
    'Rotation', -9, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
hydrostaticLegend = legend(ax3, hydrostaticHandle, ...
    {'$\sigma_1 = \sigma_2 = \sigma_3$'}, 'Location', 'none', ...
    'Box', 'off', 'Interpreter', 'latex', 'FontSize', 9);
set(hydrostaticLegend, 'Units', 'centimeters', 'Position', [1.5, 0.15, 6.0, 0.5]);

fig2 = publicationFigure(metadata.width_mm / 10, metadata.height_mm / 10);
cleanupFigure2 = onCleanup(@() close(fig2));
ax2 = axes(fig2, 'Units', 'centimeters', 'Position', [1.45, 2.15, 6.8, 6.8]);
hold(ax2, 'on');
xline(ax2, 0, '-', 'Color', [0.82, 0.82, 0.82], 'LineWidth', 0.45, 'HandleVisibility', 'off');
yline(ax2, 0, '-', 'Color', [0.82, 0.82, 0.82], 'LineWidth', 0.45, 'HandleVisibility', 'off');
% q_N = sqrt(N/2)*norm(dev_N(sigma)), as in compute_yield in the code.
% Analytic contours allow reproducible inline labels away from intersections.
creepLabels = struct('position', {}, 'rotation', {}, 'value', {}, 'color', {});
thetaCreep = linspace(0, 2 * pi, 2401);
% Trace coordinates for true-2D labels, chosen clear of the strength curves.
labelTraces = [-3.5, 1.0, -2.2];
for levelIndex = 1:numel(creepLevels)
    chi = creepLevels(levelIndex);
    creepX = chi * (cos(thetaCreep) + sin(thetaCreep) / sqrt(3));
    creepY = chi * (cos(thetaCreep) - sin(thetaCreep) / sqrt(3));
    assert(max(abs(sqrt(creepX.^2 - creepX .* creepY + creepY.^2) - chi)) < 1e-12);
    creepLabels(end + 1) = inlineCreepContour(ax2, creepX, creepY, ...
        [-chi, -chi], chi, blue);
    for branchSign = [-1, 1]
        difference = branchSign * sqrt(2) * chi;
        creepX = linspace(max(axisLimits(1), axisLimits(1) + difference), ...
            min(axisLimits(2), axisLimits(2) + difference), 1201);
        creepY = creepX - difference;
        assert(max(abs(abs(creepX - creepY) / sqrt(2) - chi)) < 1e-12);
        labelPoint = [(labelTraces(levelIndex) + difference) / 2, ...
            (labelTraces(levelIndex) - difference) / 2];
        creepLabels(end + 1) = inlineCreepContour(ax2, creepX, creepY, ...
            labelPoint, chi, red);
    end
end
[~, trueHandle] = contour(ax2, s1, s2, true2D, [0, 0], ...
    'LineColor', red, 'LineWidth', 1.25);
[~, sectionHandle] = contour(ax2, s1, s2, section3D, [0, 0], ...
    'LineColor', blue, 'LineStyle', '-', 'LineWidth', 1.25);
strengthHandle = plot(ax2, [1, 0, -ratio, 0], [0, 1, 0, -ratio], ...
    'ko', 'MarkerSize', 3.5, 'MarkerFaceColor', 'w', 'LineWidth', 0.8);
% Labels are drawn after the strength curves so coincident curves are also
% interrupted at the label (chi=1.5 on the 3-D compression envelope).
for label = creepLabels
    text(ax2, label.position(1), label.position(2), ...
        sprintf('$%.2g$', label.value), 'Interpreter', 'latex', ...
        'FontSize', 8, 'Color', label.color, 'Rotation', label.rotation, ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'BackgroundColor', 'w', 'Margin', 0.4, 'Clipping', 'on');
end
creepHandle = plot(ax2, NaN, NaN, 'k:', 'LineWidth', 0.75);
axis(ax2, 'equal'); xlim(ax2, axisLimits); ylim(ax2, axisLimits);
xlabel(ax2, '$\sigma_1 / f_{\mathrm{t}}$ [-]', 'Interpreter', 'latex', 'FontSize', 10);
ylabel(ax2, '$\sigma_2 / f_{\mathrm{t}}$ [-]', 'Interpreter', 'latex', 'FontSize', 10);
styleAxes(ax2);
legendHandle = legend(ax2, [trueHandle, sectionHandle, creepHandle, strengthHandle], ...
    {'True 2D', '3D: $\sigma_3 = 0$', 'Steady creep: $\chi$', 'Uniaxial strengths'}, ...
    'Location', 'none', 'Box', 'off', 'Interpreter', 'latex', 'FontSize', 9, ...
    'Orientation', 'horizontal', 'NumColumns', 2);
set(legendHandle, 'Units', 'centimeters', 'Position', [0.5, 0.10, 8.2, 0.85]);
% Shorter line samples keep the two legend columns apart.
legendHandle.IconColumnWidth = 20;
drawnow;
exportPublicationFigure(fig3, outputStems.threeD);
exportPublicationFigure(fig2, outputStems.twoD);
fprintf('Saved independent vector EPS/PDF figures and previews:\n  %s\n  %s\n', ...
    outputStems.threeD, outputStems.twoD);
fprintf('Each canvas: 90 x 95 mm; f_c/f_t = %.6g; maximum reference error %.3e.\n', ...
    ratio, metadata.maximum_reference_error);
end

function label = inlineCreepContour(ax, x, y, labelPoint, chi, color)
% Interrupt the dotted curve around its contour-style numeric label.
[~, index] = min(hypot(x - labelPoint(1), y - labelPoint(2)));
arcLength = [0, cumsum(hypot(diff(x), diff(y)))];
gapWidth = 0.12 + 0.055 * numel(sprintf('%.2g', chi));
gap = abs(arcLength - arcLength(index)) < gapWidth / 2;
plotX = x; plotY = y;
plotX(gap) = NaN; plotY(gap) = NaN;
plot(ax, plotX, plotY, ':', 'Color', 0.75 * color + 0.25, ...
    'LineWidth', 0.75, 'HandleVisibility', 'off');
before = max(index - 1, 1); after = min(index + 1, numel(x));
rotation = atan2d(y(after) - y(before), x(after) - x(before));
if rotation > 90, rotation = rotation - 180; end
if rotation < -90, rotation = rotation + 180; end
label = struct('position', [x(index), y(index)], 'rotation', rotation, ...
    'value', chi, 'color', color);
end

function fig = publicationFigure(widthCm, heightCm)
fig = figure('Visible', 'off', 'Color', 'w', 'Renderer', 'painters', ...
    'Units', 'centimeters', 'Position', [2, 2, widthCm, heightCm], ...
    'PaperUnits', 'centimeters', 'PaperPosition', [0, 0, widthCm, heightCm], ...
    'PaperSize', [widthCm, heightCm], 'PaperPositionMode', 'manual', 'InvertHardcopy', 'off');
end

function exportPublicationFigure(fig, outputStem)
% MATLAB substitutes unembedded Base-14 fonts. Ghostscript embeds the
% actual fonts in the submission EPS and its PDF companion, then renders
% the preview from that same EPS. All geometric content remains vector.
exportDirectory = fullfile(fileparts(fileparts(outputStem)), 'tmp', 'pdfs', 'figure_export');
if ~isfolder(exportDirectory), mkdir(exportDirectory); end
[~, baseName] = fileparts(outputStem);
rawEPS = fullfile(exportDirectory, [baseName, '_raw.eps']);
print(fig, rawEPS, '-depsc', '-painters', '-loose');
if ispc
    [status, executable] = system('where gswin64c.exe');
else
    [status, executable] = system('command -v gs');
end
assert(status == 0, 'Ghostscript is required to embed the publication fonts.');
executable = splitlines(strtrim(executable));
executable = strtrim(executable{1});
common = sprintf('"%s" -q -dBATCH -dNOPAUSE -dSAFER -dEPSCrop', executable);
fontOptions = '-dEmbedAllFonts=true -dSubsetFonts=true';
embedFonts = '-c "<</NeverEmbed []>> setdistillerparams"';
commands = {
    sprintf('%s -sDEVICE=eps2write %s -sOutputFile="%s.eps" %s -f "%s"', ...
        common, fontOptions, outputStem, embedFonts, rawEPS),
    sprintf('%s -sDEVICE=pdfwrite %s -sOutputFile="%s.pdf" %s -f "%s.eps"', ...
        common, fontOptions, outputStem, embedFonts, outputStem),
    sprintf('%s -sDEVICE=png16m -r300 -sOutputFile="%s.png" -f "%s.eps"', ...
        common, outputStem, outputStem)
};
for index = 1:numel(commands)
    [status, message] = system(commands{index});
    assert(status == 0, 'Figure export failed: %s', message);
end
end

function styleAxes(ax)
set(ax, 'TickLabelInterpreter', 'latex', 'FontSize', 9, 'LineWidth', 0.65, ...
    'TickDir', 'out', 'TickLength', [0.018, 0.018], 'Box', 'on', ...
    'XTick', -2:1:1, 'YTick', -2:1:1, 'ZTick', -2:1:1, ...
    'GridColor', [0.87, 0.87, 0.87], 'GridAlpha', 1, ...
    'GridLineStyle', ':', 'Color', 'w');
grid(ax, 'on');
end

function indicator = r2Indicator(stress, material, true2D)
% Exact polar envelope of the calibrated degree-one r2_2L potential.
% True 2D uses a 2D deviator; it is not the 3D sigma_3=0 section.
if true2D
    dimension = 2; deviatoricStrength = material.f_c;
else
    dimension = 3; deviatoricStrength = 2 * material.f_c / sqrt(3);
end
activeStress = stress(:, 1:dimension);
stressTrace = sum(activeStress, 2);
deviatoricStress = activeStress - stressTrace / dimension;
deviatoricRatio = sqrt(sum(deviatoricStress.^2, 2)) / (deviatoricStrength / sqrt(2));
volumetricStrength = material.f_t / sqrt(1 - (material.f_t / material.f_c)^2);
indicator = sqrt((stressTrace / volumetricStrength).^2 + deviatoricRatio.^2) - 1;
compression = stressTrace < 0;
indicator(compression) = max(deviatoricRatio(compression), ...
    -stressTrace(compression) / (1e6 * volumetricStrength)) - 1;
end

function metadata = validateCriterion(material)
% Compare the closed form with the current C++ finite-probe construction.
stream = RandStream('mt19937ar', 'Seed', 17);
samples = [randn(stream, 28, 3); zeros(1, 3); ones(1, 3); -ones(1, 3); ...
    1, -1, 0; 1, 1, -2] * material.f_t;
maximumError = 0;
for true2D = [false, true]
    testStress = samples;
    if true2D, testStress(:, 3) = 0; end
    exact = r2Indicator(testStress, material, true2D);
    for sample = 1:size(testStress, 1)
        reference = referenceIndicator(testStress(sample, :).', material, true2D, 2049);
        maximumError = max(maximumError, abs(reference - exact(sample)));
    end
    assert(abs(r2Indicator([material.f_t, 0, 0], material, true2D)) < 1e-12);
    assert(abs(r2Indicator([-material.f_c, 0, 0], material, true2D)) < 1e-12);
end
assert(maximumError < 3e-6, ...
    'Closed-form surface differs from the regularized finite-probe reference: %.3e.', maximumError);
metadata.maximum_reference_error = maximumError;
metadata.reference_direction_samples = 2049;
metadata.reference_stress_samples_per_dimension = size(samples, 1);
end

function indicator = referenceIndicator(stress, material, true2D, directionCount)
% Vectorized replica of compute_strength_surface_point, including its
% probe, exact polar endpoints, norm regularization and compression penalty.
if true2D
    dimension = 2; identity = [1; 1; 0];
    deviator = [0.5, -0.5, 0; -0.5, 0.5, 0; 0, 0, 0];
    strain = [material.stressToStrain2D * stress(1:2); 0];
    bulk = material.bulk2D; deviatoricStrength = material.f_c;
    fallback = [1; -1; 0] / sqrt(2);
else
    dimension = 3; identity = ones(3, 1); deviator = eye(3) - ones(3) / 3;
    strain = material.stressToStrain * stress; bulk = material.Bulk;
    deviatoricStrength = 2 * material.f_c / sqrt(3);
    fallback = [2; -1; -1] / sqrt(6);
end
strainTrace = identity.' * strain;
volDirection = identity / sqrt(dimension);
if strainTrace < 0, volDirection = -volDirection; end
devStrain = deviator * strain;
if sum(devStrain.^2) > 1e-20 * max(sum(strain.^2), 1e-20)
    devDirection = devStrain / norm(devStrain);
else
    devDirection = fallback;
end
strengthScale = max([abs(material.f_t), abs(material.f_c), 1]);
lambdaProbe = max(strengthScale / max(abs(bulk), strengthScale), 1e-8);
resistanceTolerance = 1e-12 * strengthScale * lambdaProbe;
theta = linspace(0, pi / 2, directionCount);
lambda = lambdaProbe * [cos(theta); sin(theta)];
lambda(2, 1) = 0; lambda(1, end) = 0;
eta = volDirection * lambda(1, :) + devDirection * lambda(2, :);
volumetricStrength = material.f_t / sqrt(1 - (material.f_t / material.f_c)^2);
f1 = volumetricStrength / dimension * (identity.' * eta);
etaDev = deviator * eta;
if strainTrace < 0, activeF1 = zeros(size(f1)); else, activeF1 = f1; end
resistance = sqrt(activeF1.^2 + 0.5 * deviatoricStrength^2 * ...
    sum(etaDev.^2, 1) + 1e-12) - 1e-6;
if strainTrace < 0, resistance = resistance - 1e6 * f1; end
work = stress.' * eta;
assert(all(isfinite(resistance)) && all(isfinite(work)), 'Non-finite reference evaluation.');
if any(resistance <= resistanceTolerance & work > resistanceTolerance)
    indicator = Inf;
else
    valid = resistance > resistanceTolerance;
    if any(valid), indicator = max(work(valid) ./ resistance(valid)) - 1;
    else, indicator = -1; end
end
end

