// g2o - General Graph Optimization
// Copyright (C) 2011 R. Kuemmerle, G. Grisetti, W. Burgard
//
// This file is part of g2o.
//
// g2o is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// g2o is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with g2o.  If not, see <http://www.gnu.org/licenses/>.

#ifndef G2O_ABSTRACT_PROPERTIES_WINDOW_H
#define G2O_ABSTRACT_PROPERTIES_WINDOW_H

#include <QDialog>
#include <string>
#include <vector>

#include "g2o_viewer_api.h"
#include "ui_base_properties_widget.h"

namespace g2o {
class PropertyMap;
}  // namespace g2o

/**
 * @brief Widget for displaying properties
 */
class G2O_VIEWER_API AbstractPropertiesWidget
    : public QDialog,
      public Ui::BasePropertiesWidget {
  Q_OBJECT
 public:
  explicit AbstractPropertiesWidget(QWidget* parent = nullptr);
  ~AbstractPropertiesWidget() override = default;

  virtual const g2o::PropertyMap* propertyMap() = 0;

  void on_btnApply_clicked();
  void on_btnOK_clicked();

 protected:
  std::vector<std::string> propNames_;

  virtual void updateDisplayedProperties();
  virtual void applyProperties();
  [[nodiscard]] virtual std::string humanReadablePropName(
      const std::string& propertyName) const;
};

#endif
