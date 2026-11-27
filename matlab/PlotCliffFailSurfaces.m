function metadata = PlotCliffFailSurfaces()
% Strength surfaces for the cliff case of Section 4.2, in the format of
% PlotFailSurfaces.m (90 x 95 mm canvases, normalized stresses, vector EPS/PDF).
% Run with: addpath('matlab'); PlotCliffFailSurfaces
% Panel (a): opening-only criterion in principal stress space.
% Panel (b): plane-strain sections, sigma_3 = nu (sigma_1 + sigma_2), of the
% original and the opening-only criterion, intact and after loss of cohesion.
% Both criteria share F_d = sqrt(f_t*^2 tr(eta)^2 + f_s*^2 |dev eta|^2). The
% original criterion has a volumetric and a deviatoric eigenstrain direction;
% the opening-only criterion uses the three principal dyads with nonnegative
% multipliers. Admissible stresses: gauge(sigma) <= d(phi), with the gauge the
% maximum of sigma : eta / F_d(eta) over the admissible eta.
% Panel letters and captions are supplied by LaTeX subfigures.
% Both criteria use the strengths of the cliff.

%% Material and output settings
material = struct('Poisson', 0.35, 'f_t', 0.4e6, 'f_c', 1.5e6);
assert(material.f_c > material.f_t && material.f_t > 0, ...
    'The calibrated criterion requires f_c > f_t > 0.');
% Strength scales of the three-dimensional model, squared and divided by f_t^2.
scalesOf = @(ratio) struct('ft2', 1 / (9 * (1 - ratio^(-2))), 'fs2', 2 * ratio^2 / 3);
% The original criterion always uses the two strengths of the cliff.
scalesOriginal = scalesOf(material.f_c / material.f_t);
ratio = material.f_c / material.f_t;
settings = struct('suffix', '', 'limits3', [-5, 3], 'ticks3', -4:2:2, ...
    'levels', [-2, -4], 'ringSines', [0.3, 0.6, 0.9]);
scales = scalesOf(ratio);
% Ellipsoid of the tensile branch: sigma' * inv(ellipsoid) * sigma = 1.
ellipsoid = scales.fs2 * eye(3) + (scales.ft2 - scales.fs2 / 3) * ones(3);
figureDirectory = fileparts(mfilename('fullpath'));
outputStems = struct('threeD', fullfile(figureDirectory, ['cliff_failure_surface', settings.suffix, '_3d']), ...
    'twoD', fullfile(figureDirectory, ['cliff_failure_surface', settings.suffix, '_2d']));
metadata = validateCriterion(scales, ellipsoid);
metadata.material = material;
metadata.width_mm = 90;
metadata.height_mm = 95;
metadata.output_stems = outputStems;
metadata.stress_normalization = 'f_t';
axisLimits = [-5, 3];             % panel (b)
axisLimits3 = settings.limits3;    % panel (a)
lowerLimit = axisLimits3(1);
degradationLevels = [0.5, 0.1];
planeStrain = @(s1, s2) [s1, s2, material.Poisson * (s1 + s2)];

%% Opening-only surface: tensile cap, cylinders and planes
% The admissible set is the ellipsoid extended along the negative principal
% axes. Its boundary follows from the outward normal n (nonnegative
% components) of the ellipsoid point sigma(n) = M n / sqrt(n' M n):
%   n_i > 0 for all i      cap (three opening directions)
%   n_i = 0                arc of the cap edge, extended along -e_i (two)
%   n = e_k                point of largest sigma_k, extended along the
%                          other two axes (one opening direction)
pointOf = @(normals) (ellipsoid * normals) ./ sqrt(sum(normals .* (ellipsoid * normals), 1));
unit = eye(3);
samples = linspace(0, 1, 25);
[u, v] = meshgrid(samples);
levels = [settings.levels, lowerLimit];
patches = {};       % surfaces, each n x m x 3
curves = {};        % lines, each 3 x n
for k = 1:3
    i = mod(k, 3) + 1; j = mod(k + 1, 3) + 1;
    % Cap: the octant of normals is split in three quadrilaterals around e_k.
    corners = [unit(:, k), (unit(:, k) + unit(:, i)) / 2, ones(3, 1) / 3, (unit(:, k) + unit(:, j)) / 2];
    normals = corners(:, 1) * ((1 - u(:)) .* (1 - v(:))).' + corners(:, 2) * (u(:) .* (1 - v(:))).' ...
        + corners(:, 3) * (u(:) .* v(:)).' + corners(:, 4) * ((1 - u(:)) .* v(:)).';
    cap = reshape(pointOf(normals).', [size(u), 3]);
    patches{end + 1} = cap; %#ok<AGROW>
    % Cylinder along -e_k below the cap edge between the points of largest
    % sigma_i and sigma_j.
    fraction = linspace(0, 1, 97);
    arc = pointOf(unit(:, i) * (1 - fraction) + unit(:, j) * fraction);
    cylinder = zeros(2, numel(fraction), 3);
    cylinder(1, :, :) = arc.';
    foot = arc; foot(k, :) = lowerLimit;
    cylinder(2, :, :) = foot.';
    patches{end + 1} = cylinder; %#ok<AGROW>
    curves{end + 1} = arc; %#ok<AGROW>
    for level = levels
        ring = arc; ring(k, :) = level;
        ring(:, arc(k, :) < level) = NaN;
        curves{end + 1} = ring; %#ok<AGROW>
    end
    % Plane sigma_k = const, where both other principal stresses are compressive.
    corner = pointOf(unit(:, k));
    plane = zeros(2, 2, 3);
    plane(:, :, k) = corner(k);
    plane(:, :, i) = [corner(i), corner(i); lowerLimit, lowerLimit];
    plane(:, :, j) = [corner(j), lowerLimit; corner(j), lowerLimit];
    patches{end + 1} = plane; %#ok<AGROW>
    for level = levels
        segment = repmat(corner, 1, 2);
        segment(i, :) = level; segment(j, :) = [corner(j), lowerLimit];
        curves{end + 1} = segment; %#ok<AGROW>
        segment = repmat(corner, 1, 2);
        segment(j, :) = level; segment(i, :) = [corner(i), lowerLimit];
        curves{end + 1} = segment; %#ok<AGROW>
    end
end
% Grid of the cap as in PlotFailSurfaces.m: rings of constant mean stress and
% meridians of constant Lode angle. A meridian ends on the cap edge, where a
% component of the normal vanishes, and continues along that principal axis.
basis1 = [1; -1; 0] / sqrt(2);
basis2 = [1; 1; -2] / sqrt(6);
tip = sqrt(scales.ft2);
radius = sqrt(scales.fs2);
ringAngle = linspace(0, 2 * pi, 361);
for sine = settings.ringSines
    curves{end + 1} = tip * sqrt(1 - sine^2) + radius * sine ...
        * (basis1 * cos(ringAngle) + basis2 * sin(ringAngle)); %#ok<AGROW>
end
for lode = 0:30:330
    direction = basis1 * cosd(lode) + basis2 * sind(lode);
    edgeAngle = atan(radius / (3 * tip * max(-direction)));
    polar = linspace(0, edgeAngle, 61);
    meridian = tip * cos(polar) + radius * direction * sin(polar);
    curves{end + 1} = meridian; %#ok<AGROW>
    for axisIndex = find(-direction.' > max(-direction) - 1e-9)
        foot = meridian(:, end); foot(axisIndex) = lowerLimit;
        curves{end + 1} = [meridian(:, end), foot]; %#ok<AGROW>
    end
end
surfaceError = 0;
for index = 1:numel(curves)
    points = curves{index}(:, ~any(isnan(curves{index}), 1)).';
    surfaceError = max(surfaceError, max(abs(gaugeOpening(points, scales) - 1)));
end
for index = 1:numel(patches)
    points = reshape(patches{index}, [], 3);
    surfaceError = max(surfaceError, max(abs(gaugeOpening(points, scales) - 1)));
end
assert(surfaceError < 1e-10, 'Parametric 3-D surface is inconsistent with the criterion.');
metadata.parametric_surface_error = surfaceError;

%% Plane-strain sections
axisValues = linspace(axisLimits(1), axisLimits(2), 801);
[s1, s2] = meshgrid(axisValues);
sectionStress = planeStrain(s1(:), s2(:));
sectionOriginal = reshape(gaugeOriginal(sectionStress, scalesOriginal), size(s1));
sectionOpening = reshape(gaugeOpening(sectionStress, scales), size(s1));
% Polar form of the sections: the gauge is homogeneous of degree one, so the
% surface after loss of cohesion is the intact one scaled by d(phi).
thetaOriginal = linspace(-pi, pi, 4801);
% The opening-only section is unbounded where both in-plane stresses are compressive.
thetaOpening = linspace(-pi / 2 + 1e-3, pi - 1e-3, 4801);
radiusOriginal = 1 ./ gaugeOriginal(planeStrain(cos(thetaOriginal(:)), sin(thetaOriginal(:))), scalesOriginal).';
radiusOpening = 1 ./ gaugeOpening(planeStrain(cos(thetaOpening(:)), sin(thetaOpening(:))), scales).';
metadata.plane_strain_tension = 1 / gaugeOpening(planeStrain(1, 0), scales);
metadata.plane_strain_compression_original = 1 / gaugeOriginal(planeStrain(-1, 0), scalesOriginal);
metadata.confined_tension_opening = 1 / gaugeOpening([1, -1e3, -1e3], scales);
metadata.degradation_levels = degradationLevels;

%% Final-size vector artwork
blue = [0, 114, 178] / 255;
orange = [213, 94, 0] / 255;
fig3 = publicationFigure(metadata.width_mm / 10, metadata.height_mm / 10);
cleanupFigure3 = onCleanup(@() close(fig3));
ax3 = axes(fig3, 'Units', 'centimeters', 'Position', [0.85, 2.15, 7.65, 6.8]);
hold(ax3, 'on');
viewAngles = [115, 20];
% Every outward normal of this surface has nonnegative components, so the
% whole surface faces a camera in the positive octant and no part hides
% another. Drawing in creation order (faces, then lines) is therefore exact
% and avoids the depth sorting of the vector renderer.
set(ax3, 'SortMethod', 'childorder');
% Facet edges in the face colour close the hairline seams of vector viewers.
faceColor = 0.22 * orange + 0.78;
for index = 1:numel(patches)
    surf(ax3, patches{index}(:, :, 1), patches{index}(:, :, 2), patches{index}(:, :, 3), ...
        'FaceColor', faceColor, 'EdgeColor', faceColor, 'LineWidth', 0.3, 'FaceAlpha', 1);
end
for index = 1:numel(curves)
    plot3(ax3, curves{index}(1, :), curves{index}(2, :), curves{index}(3, :), ...
        'Color', orange, 'LineWidth', 0.5);
end
% Plane-strain section, shown in panel (b).
sectionCurve = planeStrain((radiusOpening .* cos(thetaOpening)).', (radiusOpening .* sin(thetaOpening)).').';
sectionCurve(:, any(sectionCurve < lowerLimit, 1)) = NaN;
sectionHandle3 = plot3(ax3, sectionCurve(1, :), sectionCurve(2, :), sectionCurve(3, :), ...
    'k-', 'LineWidth', 0.9);
% The hydrostatic axis leaves the surface at the tip of the cap.
hydrostaticHandle = plot3(ax3, [tip, axisLimits3(2)], [tip, axisLimits3(2)], ...
    [tip, axisLimits3(2)], 'k--', 'LineWidth', 0.9);
axis(ax3, 'equal');
xlim(ax3, axisLimits3); ylim(ax3, axisLimits3); zlim(ax3, axisLimits3);
% View from the tensile octant: the tensile cap points towards the reader.
view(ax3, viewAngles);
xlabel(ax3, '$\sigma_1 / f_{\mathrm{t}}$ [-]', 'Interpreter', 'latex', 'FontSize', 10);
ylabel(ax3, '$\sigma_2 / f_{\mathrm{t}}$ [-]', 'Interpreter', 'latex', 'FontSize', 10);
zlabel(ax3, '$\sigma_3 / f_{\mathrm{t}}$ [-]', 'Interpreter', 'latex', 'FontSize', 10);
styleAxes(ax3, settings.ticks3);
% Fix the labels beside the projected axes, as in PlotFailSurfaces.m.
set(ax3.XLabel, 'Units', 'normalized', 'Position', [0.93, -0.04, 0], ...
    'Rotation', 37, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
set(ax3.YLabel, 'Units', 'normalized', 'Position', [0.31, -0.10, 0], ...
    'Rotation', -9, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
legend3 = legend(ax3, [hydrostaticHandle, sectionHandle3], ...
    {'$\sigma_1 = \sigma_2 = \sigma_3$', '$\sigma_3 = \nu(\sigma_1 + \sigma_2)$'}, ...
    'Location', 'none', 'Box', 'off', 'Interpreter', 'latex', 'FontSize', 9, ...
    'Orientation', 'horizontal', 'NumColumns', 1);
set(legend3, 'Units', 'centimeters', 'Position', [1.5, 0.10, 6.0, 0.85]);

fig2 = publicationFigure(metadata.width_mm / 10, metadata.height_mm / 10);
cleanupFigure2 = onCleanup(@() close(fig2));
ax2 = axes(fig2, 'Units', 'centimeters', 'Position', [1.45, 2.15, 6.8, 6.8]);
hold(ax2, 'on');
xline(ax2, 0, '-', 'Color', [0.82, 0.82, 0.82], 'LineWidth', 0.45, 'HandleVisibility', 'off');
yline(ax2, 0, '-', 'Color', [0.82, 0.82, 0.82], 'LineWidth', 0.45, 'HandleVisibility', 'off');
% Label positions, chosen clear of the intact surfaces and of each other.
labelOriginal = [-3.1, -0.9; -0.95, -0.95];
labelOpening = [1.3, -3.6; 0.25, -2.6];
labels = struct('position', {}, 'rotation', {}, 'value', {}, 'color', {});
for levelIndex = 1:numel(degradationLevels)
    level = degradationLevels(levelIndex);
    labels(end + 1) = inlineContour(ax2, level * radiusOriginal .* cos(thetaOriginal), ...
        level * radiusOriginal .* sin(thetaOriginal), labelOriginal(levelIndex, :), level, blue); %#ok<AGROW>
    labels(end + 1) = inlineContour(ax2, level * radiusOpening .* cos(thetaOpening), ...
        level * radiusOpening .* sin(thetaOpening), labelOpening(levelIndex, :), level, orange); %#ok<AGROW>
end
% The sections coincide on the tensile cap: the wider line stays visible below.
[~, openingHandle] = contour(ax2, s1, s2, sectionOpening, [1, 1], ...
    'LineColor', orange, 'LineStyle', '-', 'LineWidth', 2.4);
[~, originalHandle] = contour(ax2, s1, s2, sectionOriginal, [1, 1], ...
    'LineColor', blue, 'LineWidth', 1.0);
for label = labels
    text(ax2, label.position(1), label.position(2), ...
        sprintf('$%.2g$', label.value), 'Interpreter', 'latex', ...
        'FontSize', 8, 'Color', label.color, 'Rotation', label.rotation, ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'BackgroundColor', 'w', 'Margin', 0.4, 'Clipping', 'on');
end
degradedHandle = plot(ax2, NaN, NaN, 'k:', 'LineWidth', 0.75);
axis(ax2, 'equal'); xlim(ax2, axisLimits); ylim(ax2, axisLimits);
xlabel(ax2, '$\sigma_1 / f_{\mathrm{t}}$ [-]', 'Interpreter', 'latex', 'FontSize', 10);
ylabel(ax2, '$\sigma_2 / f_{\mathrm{t}}$ [-]', 'Interpreter', 'latex', 'FontSize', 10);
styleAxes(ax2, -4:2:2);
legendHandles = [originalHandle, openingHandle, degradedHandle];
legendNames = {'Two-multiplier', 'Opening-only', 'Loss of cohesion: $d(\phi)$'};
legendHandle = legend(ax2, legendHandles, legendNames, ...
    'Location', 'none', 'Box', 'off', 'Interpreter', 'latex', 'FontSize', 9, ...
    'Orientation', 'horizontal', 'NumColumns', 2);
set(legendHandle, 'Units', 'centimeters', 'Position', [0.5, 0.10, 8.2, 0.85]);
drawnow;
exportPublicationFigure(fig3, outputStems.threeD);
exportPublicationFigure(fig2, outputStems.twoD);
fprintf('Saved independent vector EPS/PDF figures and previews:\n  %s\n  %s\n', ...
    outputStems.threeD, outputStems.twoD);
fprintf(['Each canvas: 90 x 95 mm; f_c/f_t = %.6g; nu = %.3g; largest difference with the ', ...
    'dual problem %.3e.\n'], ratio, material.Poisson, metadata.maximum_reference_error);
fprintf(['Strengths / f_t: plane-strain tension %.4f, plane-strain compression (original) %.4f, ', ...
    'tension under strong lateral compression (opening-only) %.4f.\n'], ...
    metadata.plane_strain_tension, metadata.plane_strain_compression_original, ...
    metadata.confined_tension_opening);
end

function label = inlineContour(ax, x, y, labelPoint, value, color)
% Interrupt the dotted curve around its contour-style numeric label.
[~, index] = min(hypot(x - labelPoint(1), y - labelPoint(2)));
arcLength = [0, cumsum(hypot(diff(x), diff(y)))];
gapWidth = 2 * (0.12 + 0.055 * numel(sprintf('%.2g', value)));
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
    'value', value, 'color', color);
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

function styleAxes(ax, ticks)
set(ax, 'TickLabelInterpreter', 'latex', 'FontSize', 9, 'LineWidth', 0.65, ...
    'TickDir', 'out', 'TickLength', [0.018, 0.018], 'Box', 'on', ...
    'XTick', ticks, 'YTick', ticks, 'ZTick', ticks, ...
    'GridColor', [0.87, 0.87, 0.87], 'GridAlpha', 1, ...
    'GridLineStyle', ':', 'Color', 'w');
grid(ax, 'on');
end

function gauge = gaugeOriginal(stress, scales)
% Original criterion (principal stresses / f_t, one state per row): the
% ellipsoid for tr(sigma) >= 0, the deviatoric limit for tr(sigma) < 0.
stressTrace = sum(stress, 2);
deviatoric = max(sum(stress.^2, 2) - stressTrace.^2 / 3, 0);
gauge = sqrt(max(stressTrace, 0).^2 / (9 * scales.ft2) + deviatoric / scales.fs2);
end

function gauge = gaugeOpening(stress, scales)
% Opening-only criterion: largest value over the sets of opening directions
% whose stationary multipliers are all nonnegative (zero without a tensile
% principal stress).
gauge = zeros(size(stress, 1), 1);
for set = 1:7
    active = bitget(set, 1:3) == 1;
    count = nnz(active);
    activeStress = stress(:, active);
    meanStress = mean(activeStress, 2);
    multipliers = (activeStress - meanStress) / scales.fs2 ...
        + meanStress / (scales.fs2 + count * (scales.ft2 - scales.fs2 / 3));
    value = sqrt(max(sum(activeStress .* multipliers, 2), 0));
    admissible = all(multipliers >= 0, 2);
    gauge(admissible) = max(gauge(admissible), value(admissible));
end
end

function metadata = validateCriterion(scales, ellipsoid)
% Compare the closed form with the dual problem: the smallest ellipsoid
% gauge of sigma + mu over mu >= 0 (a non-negative least-squares problem).
stream = RandStream('mt19937ar', 'Seed', 17);
samples = [randn(stream, 400, 3) .* (0.2 + 4 * rand(stream, 400, 1)); zeros(1, 3); ...
    ones(1, 3); -ones(1, 3); 1, -1, 0; 1, 1, -2; 1, 0, 0; 1, -50, -50];
factor = chol(inv(ellipsoid));      % ellipsoid gauge of t: norm(factor * t)
exact = gaugeOpening(samples, scales);
maximumError = 0;
for sample = 1:size(samples, 1)
    [~, squaredResidual] = lsqnonneg(factor, -factor * samples(sample, :).');
    maximumError = max(maximumError, abs(sqrt(squaredResidual) - exact(sample)));
end
assert(maximumError < 1e-8, ...
    'Closed-form surface differs from the dual problem: %.3e.', maximumError);
% f_t stays the uniaxial tensile strength; the original criterion keeps f_c.
assert(abs(gaugeOpening([1, 0, 0], scales) - 1) < 1e-12);
assert(abs(gaugeOriginal([1, 0, 0], scales) - 1) < 1e-12);
assert(abs(gaugeOriginal([-sqrt(1.5 * scales.fs2), 0, 0], scales) - 1) < 1e-12);
metadata.maximum_reference_error = maximumError;
metadata.reference_stress_samples = size(samples, 1);
end
