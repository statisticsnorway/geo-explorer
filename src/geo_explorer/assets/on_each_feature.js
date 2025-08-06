window.onEachFeatureToggleHighlight = Object.assign({}, window.dashExtensions, {
    default: {
        yellowIfHighlighted: function (feature, layer) {
            const geomType = feature.geometry.type;

            // Store original style ON THE LAYER
            layer._originalStyle = {
                color: layer.options.color,
                fillColor: layer.options.fillColor,
                weight: layer.options.weight,
                fillOpacity: layer.options.fillOpacity
            };

            layer.on("click", function () {
                // Highlight yellow
                let start = null;
                let duration = 700; // ms
                let initialFillOpacity = layer.options.fillOpacity;
                let initialWeight = layer.options.weight;

                function animateFade(timestamp) {
                    if (!start) start = timestamp;
                    let elapsed = timestamp - start;
                    let progress = Math.min(elapsed / duration, 1);

                    // Interpolate fillOpacity and weight
                    let fillOpacity = 0.9 * (1 - progress) + initialFillOpacity * progress;
                    let weight = 3 * (1 - progress) + initialWeight * progress;

                    if (geomType === "LineString" || geomType === "MultiLineString" || geomType === "LinearRing") {
                        layer.setStyle({
                            fillColor: "#ffff00",
                            color: "#ffff00",
                            weight: weight,
                            fillOpacity: fillOpacity
                        });
                    } else {
                        layer.setStyle({
                            fillColor: "#ffff00",
                            color: "#000000",
                            weight: weight,
                            fillOpacity: fillOpacity
                        });
                    }

                    if (progress < 1) {
                        window.requestAnimationFrame(animateFade);
                    } else {
                        layer.setStyle(layer._originalStyle);
                    }
                }

                window.requestAnimationFrame(animateFade);
            });
        },
        pointToLayerCircle: function(feature, latlng, context) {
            const {min, max, colorscale, circleOptions, colorProp} = context.hideout;
            const csc = chroma.scale(colorscale).domain([min, max]);
            circleOptions.fillColor = csc(feature.properties[colorProp]);
            return L.circleMarker(latlng, circleOptions);
        }
    }
});