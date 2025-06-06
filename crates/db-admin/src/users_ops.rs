use super::{AdminDatabase, User};

impl AdminDatabase {
    pub async fn list_users(&self) -> Result<Vec<User>, crate::Error> {
        let conn = self.conn()?;

        let mut rows = conn.query("SELECT * FROM users", ()).await.unwrap();
        let mut users = Vec::new();

        while let Some(row) = rows.next().await.unwrap() {
            let user: User = libsql::de::from_row(&row).unwrap();
            users.push(user);
        }

        Ok(users)
    }

    pub async fn upsert_user(&self, user: User) -> Result<User, crate::Error> {
        let conn = self.conn()?;

        let mut rows = conn
            .query(
                "INSERT INTO users (
                    id,
                    account_id,
                    human_id,
                    timestamp,
                    clerk_user_id
                ) VALUES (?, ?, ?, ?, ?) 
                ON CONFLICT (id) DO UPDATE SET
                    account_id = excluded.account_id,
                    human_id = excluded.human_id,
                    timestamp = excluded.timestamp,
                    clerk_user_id = excluded.clerk_user_id
                RETURNING *",
                vec![
                    user.id,
                    user.account_id,
                    user.human_id,
                    user.timestamp.to_rfc3339(),
                    user.clerk_user_id,
                ],
            )
            .await?;

        let row = rows.next().await.unwrap().unwrap();
        let user: User = libsql::de::from_row(&row).unwrap();
        Ok(user)
    }

    pub async fn get_user_by_clerk_user_id(
        &self,
        clerk_user_id: impl AsRef<str>,
    ) -> Result<Option<User>, crate::Error> {
        let conn = self.conn()?;

        let mut rows = conn
            .query(
                "SELECT * FROM users WHERE clerk_user_id = ?",
                vec![clerk_user_id.as_ref()],
            )
            .await?;

        match rows.next().await.unwrap() {
            None => Ok(None),
            Some(row) => {
                let user: User = libsql::de::from_row(&row).unwrap();
                Ok(Some(user))
            }
        }
    }

    pub async fn get_user_by_device_api_key(
        &self,
        api_key: impl AsRef<str>,
    ) -> Result<Option<User>, crate::Error> {
        let conn = self.conn()?;

        let mut rows = conn
            .query(
                "SELECT users.* FROM users
                JOIN devices ON devices.user_id = users.id
                WHERE devices.api_key = ?",
                vec![api_key.as_ref()],
            )
            .await?;

        match rows.next().await.unwrap() {
            None => Ok(None),
            Some(row) => {
                let user: User = libsql::de::from_row(&row).unwrap();
                Ok(Some(user))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{tests::setup_db, Account, Device};

    #[tokio::test]
    async fn test_create_list_get_user() {
        let db = setup_db().await;

        let account = db
            .upsert_account(Account {
                id: uuid::Uuid::new_v4().to_string(),
                turso_db_name: "yujonglee".to_string(),
                clerk_org_id: Some("org_1".to_string()),
            })
            .await
            .unwrap();

        let user = db
            .upsert_user(User {
                id: uuid::Uuid::new_v4().to_string(),
                account_id: account.id.clone(),
                human_id: uuid::Uuid::new_v4().to_string(),
                timestamp: chrono::Utc::now(),
                clerk_user_id: "21".to_string(),
            })
            .await
            .unwrap();
        assert_eq!(user.clerk_user_id, "21".to_string());

        let users = db.list_users().await.unwrap();
        assert_eq!(users.len(), 1);

        let _user = db
            .get_user_by_clerk_user_id("21".to_string())
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn test_create_list_get_device() {
        let db = setup_db().await;

        let account = db
            .upsert_account(Account {
                id: uuid::Uuid::new_v4().to_string(),
                turso_db_name: "yujonglee".to_string(),
                clerk_org_id: Some("org_1".to_string()),
            })
            .await
            .unwrap();

        let user = db
            .upsert_user(User {
                id: uuid::Uuid::new_v4().to_string(),
                account_id: account.id.clone(),
                human_id: uuid::Uuid::new_v4().to_string(),
                timestamp: chrono::Utc::now(),
                clerk_user_id: "21".to_string(),
            })
            .await
            .unwrap();

        let device = db
            .upsert_device(Device {
                id: uuid::Uuid::new_v4().to_string(),
                timestamp: chrono::Utc::now(),
                user_id: user.id.clone(),
                fingerprint: "fingerprint".to_string(),
                api_key: "key".to_string(),
            })
            .await
            .unwrap();

        assert_eq!(device.user_id, user.id);
    }

    #[tokio::test]
    async fn test_get_user_by_device_api_key() {
        let db = setup_db().await;

        let account = db
            .upsert_account(Account {
                id: uuid::Uuid::new_v4().to_string(),
                turso_db_name: "yujonglee".to_string(),
                clerk_org_id: Some("org_1".to_string()),
            })
            .await
            .unwrap();

        let user_1 = db
            .upsert_user(User {
                id: uuid::Uuid::new_v4().to_string(),
                account_id: account.id.clone(),
                human_id: uuid::Uuid::new_v4().to_string(),
                timestamp: chrono::Utc::now(),
                clerk_user_id: "21".to_string(),
            })
            .await
            .unwrap();

        let device = db
            .upsert_device(Device {
                id: uuid::Uuid::new_v4().to_string(),
                timestamp: chrono::Utc::now(),
                user_id: user_1.id.clone(),
                fingerprint: "fingerprint".to_string(),
                api_key: "key".to_string(),
            })
            .await
            .unwrap();

        let user_2 = db
            .get_user_by_device_api_key(device.api_key)
            .await
            .unwrap()
            .unwrap();

        assert_eq!(user_1.id, user_2.id);
    }
}
