"""Vertex service-account key parsing: PKCS#1 and PKCS#8 must both work."""

from pi.ai.providers.vertex import load_private_key, service_account_jwt

PKCS1_PEM = """-----BEGIN RSA PRIVATE KEY-----
MIIEpgIBAAKCAQEAwI7m9JBsWWoc5IwOiGGYp6344y9prL4/bxrIRfpVC39EbUvv
r2EztSdzMPIj2csJnFZI9qCjB15poGk+0mqvfWfyA9Bigw7NzivKjOBls/G8kyVi
cbA5d3PQZtiOehdrZWLrF4V5t2deYIbBB/D6LSB79npHRkjZKTJL7IGhNLfMCx+A
FADWP5WWz//LS+exhH0SFyYzxytYeMDvs6OYlE7E5pOFAN5k8XRFqmq0v8zs/OkQ
6q6XeRzeb/7GtL8+mrGFUlh30aWm3QdHc5MQ6OfuasVxh2KCGD1pjK7HQPbnWKW3
KVBu/kS4ylponKU/iuPoDJEzRY99BjXJwWfOYwIDAQABAoIBAQCVsAD16TdxIr4n
JIE/n67N2r35zae2c4ZYVleNnpZyA5Mr67X9ffB2SJ6RVApuihmEgMbjgeh6F/jA
jFAbvneBY0aNlHagLufp97zTWrV09tyC/rLScg2aQFunN306h+4iuIzMkiQWYLLL
ErYH+FpirjfvWQsDJio6jjDbP8ITedpMK571d1nGENcpAkxkmZEGNKanxCDVqCGl
pwrU5LtCkixgWQNqdO4yRNalZSBz4mtZoHbRnis2p3S9COp8pHgGDcz8v2KKIwVw
LPN3t1P4daHyq86PwqqiM5TNJwgPGiKzv3m1Z6/krD1kSrAnpF2VIETqZEOxKoPV
3PUKQMGxAoGBAPfJActgRifHDiKOrFQS1URsjZEwL7QfkQiSN+7wcaRPh4lAPT0b
ZjlamhBfp48ubbt/fGWwXFFGsiZOCBfgg4oo2T6AjyNP76G/m+8HTLWJeY6AcBOc
f+BU9BCm8ezCAbZCdnCupMdIl6pwVoCH0sgjTym3AUZ0gCm9owDPocQnAoGBAMbx
LMZk94mHzUOqNYUoVtPN3qfX4lh90heh6UnI+droBVtBAeS0+iRMDl9UVLWq7JvT
xhcaQeOF98/wNWBXhQ4v8uS66T3yeP1FH9UjaMyiv0D9jM5EXO/JGrNprE4rSq/W
BoaffHbqcncO89WohbMQKHdUI6XByHVw6NZa2x1lAoGBAPQj9IWrtVQqGC5Kf1dT
OxjxEEeWv9ntJFzanQvrkz0/Mx8z0WLtJTqrn5M0fIdiSrJseWhEbaGTU6LUTbsg
9tphRzQDPYWglE692WmZuqY/7ypOOJ/KGmExs0zZAROzUDaBo+uA2FCWoQFJMKHx
seiKicxjSXle/oyiEgKRkiUPAoGBALfUviopvG9SlMN08fW0m98mOhvgqmBZC50Y
2Wh8FR1KD0EOpgA/l3GY8iL22zvv0KoBm+dXNRFYhvCXp8r5tJKyjgD8EnFSsKhg
ipS6HqoRrScR5F3p3/MXYtvsxvKIoIAxBmC52RpNdC4+G+UPz1Pyl2IE1tfhZCnZ
ht4koJrdAoGBAO99eOzXx4MV8Wrqt4WpCSlVhqbDWjDgInWjiIGAB7ZQRmvfadE8
IUPEN9e6yP8QFkeGgmbR6TfNRduPdBWvgWcTsQFlRj9RHFj0JdQ9bKFTwsOHb+c4
sg4wvx7OO+7mW07dlLBcJ6mQoe6HpZew6OVHZ6tm1OvUbVBqydTCMdzr
-----END RSA PRIVATE KEY-----
"""

PKCS8_PEM = """-----BEGIN PRIVATE KEY-----
MIIEwAIBADANBgkqhkiG9w0BAQEFAASCBKowggSmAgEAAoIBAQDAjub0kGxZahzk
jA6IYZinrfjjL2msvj9vGshF+lULf0RtS++vYTO1J3Mw8iPZywmcVkj2oKMHXmmg
aT7Saq99Z/ID0GKDDs3OK8qM4GWz8byTJWJxsDl3c9Bm2I56F2tlYusXhXm3Z15g
hsEH8PotIHv2ekdGSNkpMkvsgaE0t8wLH4AUANY/lZbP/8tL57GEfRIXJjPHK1h4
wO+zo5iUTsTmk4UA3mTxdEWqarS/zOz86RDqrpd5HN5v/sa0vz6asYVSWHfRpabd
B0dzkxDo5+5qxXGHYoIYPWmMrsdA9udYpbcpUG7+RLjKWmicpT+K4+gMkTNFj30G
NcnBZ85jAgMBAAECggEBAJWwAPXpN3EivickgT+frs3avfnNp7ZzhlhWV42elnID
kyvrtf198HZInpFUCm6KGYSAxuOB6HoX+MCMUBu+d4FjRo2UdqAu5+n3vNNatXT2
3IL+stJyDZpAW6c3fTqH7iK4jMySJBZgsssStgf4WmKuN+9ZCwMmKjqOMNs/whN5
2kwrnvV3WcYQ1ykCTGSZkQY0pqfEINWoIaWnCtTku0KSLGBZA2p07jJE1qVlIHPi
a1mgdtGeKzandL0I6nykeAYNzPy/YoojBXAs83e3U/h1ofKrzo/CqqIzlM0nCA8a
IrO/ebVnr+SsPWRKsCekXZUgROpkQ7Eqg9Xc9QpAwbECgYEA98kBy2BGJ8cOIo6s
VBLVRGyNkTAvtB+RCJI37vBxpE+HiUA9PRtmOVqaEF+njy5tu398ZbBcUUayJk4I
F+CDiijZPoCPI0/vob+b7wdMtYl5joBwE5x/4FT0EKbx7MIBtkJ2cK6kx0iXqnBW
gIfSyCNPKbcBRnSAKb2jAM+hxCcCgYEAxvEsxmT3iYfNQ6o1hShW083ep9fiWH3S
F6HpScj52ugFW0EB5LT6JEwOX1RUtarsm9PGFxpB44X3z/A1YFeFDi/y5LrpPfJ4
/UUf1SNozKK/QP2MzkRc78kas2msTitKr9YGhp98dupydw7z1aiFsxAod1QjpcHI
dXDo1lrbHWUCgYEA9CP0hau1VCoYLkp/V1M7GPEQR5a/2e0kXNqdC+uTPT8zHzPR
Yu0lOqufkzR8h2JKsmx5aERtoZNTotRNuyD22mFHNAM9haCUTr3ZaZm6pj/vKk44
n8oaYTGzTNkBE7NQNoGj64DYUJahAUkwofGx6IqJzGNJeV7+jKISApGSJQ8CgYEA
t9S+Kim8b1KUw3Tx9bSb3yY6G+CqYFkLnRjZaHwVHUoPQQ6mAD+XcZjyIvbbO+/Q
qgGb51c1EViG8Jenyvm0krKOAPwScVKwqGCKlLoeqhGtJxHkXenf8xdi2+zG8oig
gDEGYLnZGk10Lj4b5Q/PU/KXYgTW1+FkKdmG3iSgmt0CgYEA73147NfHgxXxauq3
hakJKVWGpsNaMOAidaOIgYAHtlBGa99p0TwhQ8Q317rI/xAWR4aCZtHpN81F2490
Fa+BZxOxAWVGP1EcWPQl1D1soVPCw4dv5ziyDjC/Hs477uZbTt2UsFwnqZCh7oel
l7Do5Udnq2bU69RtUGrJ1MIx3Os=
-----END PRIVATE KEY-----
"""


def test_pkcs1_and_pkcs8_parse_to_the_same_key():
    pkcs1 = load_private_key(PKCS1_PEM)
    pkcs8 = load_private_key(PKCS8_PEM)
    assert pkcs1 == pkcs8
    n, d = pkcs1
    assert n.bit_length() == 2048
    assert d > 1


def test_service_account_jwt_signs_with_pkcs8_key():
    account = {
        "client_email": "test@example.iam.gserviceaccount.com",
        "private_key": PKCS8_PEM,
    }
    token = service_account_jwt(account, now=1700000000)
    header, claims, signature = token.split(".")
    assert header and claims
    # A 2048-bit RSA signature is 256 bytes: 342 base64url characters.
    assert len(signature) == 342
